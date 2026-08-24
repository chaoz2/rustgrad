use super::*;
use crate::models::transformer::{
    LLAMA_SIMPLE_CHAT_TEMPLATE, LlamaChatMessage, LlamaChatRole, LlamaChatTemplate,
    LlamaNativeGenerator, LlamaSampling,
    model_tests::{VOCAB, make_model, serialized_model_with_template},
};

fn greedy(tokens: usize) -> LlamaServingGenerationConfig {
    LlamaServingGenerationConfig::new(tokens, LlamaServingSampling::Greedy)
}

fn finish(scheduler: &mut LlamaServingScheduler<'_>) {
    while scheduler.pending() != 0 {
        assert!(!scheduler.step().unwrap().is_empty());
    }
}

#[test]
fn continuous_arrival_prefix_hit_and_native_results_match_independent_runs() {
    let (model, tokenizer, _) = make_model(10);
    let mut scheduler = LlamaServingScheduler::new(
        &model,
        &tokenizer,
        LlamaServingConfig::new(2, 8, 1 << 20).unwrap(),
    );
    let first_prompt = vec![3, 4, 5];
    let first = scheduler
        .submit_ids(first_prompt.clone(), greedy(2))
        .unwrap();
    let first_events = scheduler.step().unwrap();
    assert_eq!(first_events.len(), 1);
    let compiled = scheduler.native_compile_cache_len();
    assert!(compiled > 0);

    let overlapping_prompt = vec![3, 4, 5, 6];
    let second = scheduler
        .submit_ids(overlapping_prompt.clone(), greedy(2))
        .unwrap();
    let mixed = scheduler.step().unwrap();
    assert!(scheduler.native_compile_cache_len() >= compiled);
    assert_eq!(
        mixed
            .iter()
            .map(LlamaTokenEvent::request_id)
            .collect::<Vec<_>>(),
        vec![first, second]
    );
    finish(&mut scheduler);
    assert!(scheduler.prefix_stats().hits >= 1);

    let expected_first = LlamaNativeGenerator::new(&model, &tokenizer)
        .generate_ids(&first_prompt, 2, LlamaSampling::Greedy)
        .unwrap();
    let expected_second = LlamaNativeGenerator::new(&model, &tokenizer)
        .generate_ids(&overlapping_prompt, 2, LlamaSampling::Greedy)
        .unwrap();
    assert_eq!(
        scheduler.result(first).unwrap().generated_ids(),
        expected_first.generated_ids()
    );
    assert_eq!(
        scheduler.result(second).unwrap().generated_ids(),
        expected_second.generated_ids()
    );
}

#[test]
fn shared_prefix_rows_diverge_without_mutating_each_other() {
    let (model, tokenizer, _) = make_model(12);
    let mut scheduler = LlamaServingScheduler::new(
        &model,
        &tokenizer,
        LlamaServingConfig::new(2, 8, 1 << 20).unwrap(),
    );
    let seed = scheduler.submit_ids(vec![3, 4], greedy(1)).unwrap();
    finish(&mut scheduler);
    assert!(scheduler.result(seed).is_some());

    let a_prompt = vec![3, 4, 5];
    let b_prompt = vec![3, 4, 6];
    let a = scheduler.submit_ids(a_prompt.clone(), greedy(2)).unwrap();
    let b = scheduler.submit_ids(b_prompt.clone(), greedy(2)).unwrap();
    finish(&mut scheduler);
    assert!(scheduler.prefix_stats().hits >= 2);
    for (id, prompt) in [(a, a_prompt), (b, b_prompt)] {
        let expected = LlamaNativeGenerator::new(&model, &tokenizer)
            .generate_ids(&prompt, 2, LlamaSampling::Greedy)
            .unwrap();
        assert_eq!(
            scheduler.result(id).unwrap().generated_ids(),
            expected.generated_ids()
        );
    }
}

#[test]
fn staged_native_failure_rolls_back_selected_requests_and_prefix_accounting() {
    let (model, tokenizer, _) = make_model(8);
    let mut scheduler = LlamaServingScheduler::new(
        &model,
        &tokenizer,
        LlamaServingConfig::new(1, 8, 1 << 20).unwrap(),
    );
    let first = scheduler.submit_ids(vec![3, 4], greedy(2)).unwrap();
    let second = scheduler.submit_ids(vec![5], greedy(2)).unwrap();
    let before = scheduler.prefix_stats();
    scheduler.inject_stage_failure(Some(0));
    assert!(matches!(
        scheduler.step(),
        Err(LlamaServingError::Native(
            LlamaNativeError::InjectedStageFailure(0)
        ))
    ));
    assert_eq!(scheduler.status(first), Some(LlamaRequestStatus::Queued));
    assert_eq!(scheduler.status(second), Some(LlamaRequestStatus::Queued));
    assert_eq!(scheduler.prefix_stats(), before);

    scheduler.inject_stage_failure(None);
    finish(&mut scheduler);
    assert!(scheduler.result(first).is_some());
    assert!(scheduler.result(second).is_some());
    assert_eq!(
        scheduler
            .completed_results()
            .take(2)
            .map(LlamaServingResult::request_id)
            .collect::<Vec<_>>(),
        vec![first, second]
    );
}

#[test]
fn per_request_gumbel_tape_matches_single_native_generation() {
    let (model, tokenizer, _) = make_model(8);
    let uniforms = (0..2 * VOCAB)
        .map(|index| 0.07 + index as f32 * 0.035)
        .collect::<Vec<_>>();
    let prompt = vec![3, 5];
    let expected = LlamaNativeGenerator::new(&model, &tokenizer)
        .generate_ids(
            &prompt,
            2,
            LlamaSampling::GumbelMax {
                temperature: 0.8,
                uniforms: &uniforms,
            },
        )
        .unwrap();
    let mut scheduler = LlamaServingScheduler::new(
        &model,
        &tokenizer,
        LlamaServingConfig::new(1, 4, 1 << 20).unwrap(),
    );
    let id = scheduler
        .submit_ids(
            prompt,
            LlamaServingGenerationConfig::new(
                2,
                LlamaServingSampling::GumbelMax {
                    temperature: 0.8,
                    uniforms,
                },
            ),
        )
        .unwrap();
    finish(&mut scheduler);
    assert_eq!(
        scheduler.result(id).unwrap().generated_ids(),
        expected.generated_ids()
    );
    assert_eq!(scheduler.result(id).unwrap().decoded(), expected.decoded());
}

#[test]
fn text_and_checked_chat_admission_use_the_public_tokenizer_path() {
    let bytes = serialized_model_with_template(32, Some(LLAMA_SIMPLE_CHAT_TEMPLATE));
    let file = crate::gguf::read_gguf(&bytes).unwrap();
    let (model, tokenizer) = LlamaModel::from_gguf(&file).unwrap();
    let template = LlamaChatTemplate::from_gguf(&file).unwrap();
    let messages = [LlamaChatMessage::new(LlamaChatRole::User, "a").unwrap()];
    let rendered = template.render(&tokenizer, &messages, true).unwrap();
    let expected_prompt = tokenizer.encode(&rendered).unwrap();
    let mut scheduler = LlamaServingScheduler::new(
        &model,
        &tokenizer,
        LlamaServingConfig::new(2, 4, 1 << 20).unwrap(),
    );
    let text = scheduler.submit_text("abc", greedy(1)).unwrap();
    let chat = scheduler
        .submit_chat(template, &messages, greedy(1))
        .unwrap();
    finish(&mut scheduler);
    assert_eq!(
        scheduler.result(text).unwrap().prompt_ids(),
        tokenizer.encode("abc").unwrap()
    );
    assert_eq!(
        scheduler.result(chat).unwrap().prompt_ids(),
        expected_prompt
    );
}

#[test]
fn stop_context_removal_lru_and_stale_entries_are_explicit() {
    let (model, tokenizer, _) = make_model(8);
    let mut scheduler = LlamaServingScheduler::new(
        &model,
        &tokenizer,
        LlamaServingConfig::new(1, 1, 1 << 20).unwrap(),
    );
    let first = scheduler.submit_ids(vec![3], greedy(1)).unwrap();
    finish(&mut scheduler);
    assert_eq!(scheduler.prefix_stats().entries, 1);
    assert!(scheduler.prefix_stats().bytes > 0);
    assert!(scheduler.prefix_stats().bytes <= 1 << 20);
    let second = scheduler.submit_ids(vec![4], greedy(1)).unwrap();
    finish(&mut scheduler);
    assert!(scheduler.prefix_stats().evictions >= 1);
    assert_eq!(scheduler.prefix_stats().entries, 1);
    assert!(scheduler.result(first).is_some());
    assert!(scheduler.result(second).is_some());

    scheduler.make_prefixes_stale();
    let stale = scheduler.submit_ids(vec![4, 5], greedy(1)).unwrap();
    finish(&mut scheduler);
    assert!(scheduler.prefix_stats().stale_rejections >= 1);
    assert!(scheduler.result(stale).is_some());

    let mut stop_tape = vec![1e-12; VOCAB];
    stop_tape[model.config().token_ids().eos() as usize] = 0.999_999;
    let stop = scheduler
        .submit_ids(
            vec![3],
            LlamaServingGenerationConfig::new(
                2,
                LlamaServingSampling::GumbelMax {
                    temperature: 1.0,
                    uniforms: [stop_tape.clone(), stop_tape].concat(),
                },
            ),
        )
        .unwrap();
    finish(&mut scheduler);
    assert!(scheduler.result(stop).unwrap().stopped());
    assert_eq!(scheduler.result(stop).unwrap().generated_ids().len(), 1);

    let removable = scheduler.submit_ids(vec![3], greedy(1)).unwrap();
    assert!(scheduler.remove(removable));
    assert_eq!(scheduler.status(removable), None);
    assert!(matches!(
        scheduler.submit_ids(vec![3; 8], greedy(1)),
        Err(LlamaServingError::ContextLength {
            requested: 9,
            maximum: 8
        })
    ));

    let generation = scheduler.prefix_stats().generation;
    let (other_model, other_tokenizer, _) = make_model(10);
    assert!(scheduler.rebind(&other_model, &other_tokenizer).unwrap());
    assert_eq!(scheduler.prefix_stats().entries, 0);
    assert_eq!(scheduler.prefix_stats().generation, generation + 1);
}
