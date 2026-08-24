use super::{
    LLAMA_SIMPLE_CHAT_TEMPLATE, LlamaBatchGenerator, LlamaBatchNativeGenerator, LlamaBatchSampling,
    LlamaChatMessage, LlamaChatRole, LlamaChatTemplate, LlamaGenerator, LlamaNativeError,
    LlamaNativeGenerationError, LlamaNativeGenerator, LlamaNativeStageKind, LlamaSampling,
    generation::select_last,
    model_tests::{VOCAB, make_model, serialized_model_with_template},
};
use crate::ItemBackend;

fn forced_tape(tokens: &[usize]) -> Vec<f32> {
    let mut tape = vec![1e-9; tokens.len() * VOCAB];
    for (step, token) in tokens.iter().copied().enumerate() {
        tape[step * VOCAB + token] = 0.9;
    }
    tape
}

#[test]
fn native_generation_matches_direct_selection_and_reuses_decode_artifacts() {
    let (model, tokenizer, _) = make_model(8);
    let greedy = model
        .generate_native(&tokenizer, "a", 2, LlamaSampling::Greedy)
        .unwrap();
    let direct = LlamaGenerator::new(&model, &tokenizer)
        .generate_text("a", 2, LlamaSampling::Greedy)
        .unwrap();
    assert_eq!(greedy.prompt_ids(), direct.prompt_ids());
    assert_eq!(greedy.generated_ids(), direct.generated_ids());
    assert_eq!(greedy.decoded(), direct.decoded());

    let tape = forced_tape(&[3, 4, 5]);
    let sampling = LlamaSampling::GumbelMax {
        temperature: 1e6,
        uniforms: &tape,
    };
    let prompt = tokenizer.encode("a").unwrap();
    let mut prompt_with_bos = prompt.clone();
    if let Some(bos) = model.config().token_ids().bos() {
        prompt_with_bos.insert(0, bos);
    }
    let first_logits = model.forward(&prompt_with_bos).unwrap();
    assert_eq!(select_last(&first_logits, sampling, 0, VOCAB).unwrap(), 3);

    let mut native_generator = LlamaNativeGenerator::new(&model, &tokenizer);
    let native = native_generator.generate_text("a", 3, sampling).unwrap();
    let direct = LlamaGenerator::new(&model, &tokenizer)
        .generate_text("a", 3, sampling)
        .unwrap();
    assert_eq!(native.generated_ids(), &[3, 4, 5]);
    assert_eq!(native.generated_ids(), direct.generated_ids());
    assert_eq!(native.decoded(), direct.decoded());
    assert_eq!(native_generator.cache_len(), 3);
    assert_eq!(native.trace().len(), 3);
    assert!(
        native
            .trace()
            .iter()
            .flat_map(|step| step.stages())
            .filter(|stage| stage.kind == LlamaNativeStageKind::NativeSchedule)
            .flat_map(|stage| &stage.items)
            .all(|item| item.backend == ItemBackend::NativeJit)
    );
    assert!(
        native
            .trace()
            .iter()
            .flat_map(|step| step.stages())
            .flat_map(|stage| &stage.items)
            .any(|item| item.cache_hit)
    );
}

#[test]
fn native_fixed_batch_generation_matches_direct_rows_and_stop_semantics() {
    let (model, tokenizer, _) = make_model(8);
    let mut tape = vec![1e-9; 2 * 2 * VOCAB];
    tape[1] = 0.9;
    tape[VOCAB + 3] = 0.9;
    tape[3 * VOCAB + 2] = 0.9;
    let sampling = LlamaBatchSampling::GumbelMax {
        temperature: 1e6,
        uniforms: &tape,
    };
    let direct = LlamaBatchGenerator::new(&model, &tokenizer, 2)
        .unwrap()
        .generate_texts(&["a", "bc"], 2, sampling)
        .unwrap();
    let mut native_generator = LlamaBatchNativeGenerator::new(&model, &tokenizer, 2).unwrap();
    let native = native_generator
        .generate_texts(&["a", "bc"], 2, sampling)
        .unwrap();
    for (actual, expected) in native.sequences().iter().zip(direct.sequences()) {
        assert_eq!(actual.prompt_ids(), expected.prompt_ids());
        assert_eq!(actual.generated_ids(), expected.generated_ids());
        assert_eq!(actual.decoded(), expected.decoded());
        assert_eq!(actual.stopped(), expected.stopped());
    }
    assert_eq!(native.sequences()[0].generated_ids(), &[1]);
    assert_eq!(native.sequences()[1].generated_ids(), &[3, 2]);
    assert_eq!(native_generator.cache_lengths(), &[1, 3]);
    assert_eq!(native.trace()[0].input_lengths(), &[1, 2]);
    assert_eq!(native.trace()[1].input_lengths(), &[0, 1]);
    assert_eq!(native.trace()[1].cache_before(), &[1, 2]);
    assert_eq!(native.trace()[1].cache_after(), &[1, 3]);
}

#[test]
fn checked_chat_prompt_runs_through_native_generation() {
    let bytes = serialized_model_with_template(16, Some(LLAMA_SIMPLE_CHAT_TEMPLATE));
    let file = crate::gguf::read_gguf(&bytes).unwrap();
    let (model, tokenizer) = super::LlamaModel::from_gguf(&file).unwrap();
    let template = LlamaChatTemplate::from_gguf(&file).unwrap();
    let messages = [LlamaChatMessage::new(LlamaChatRole::User, "a").unwrap()];
    let rendered = template.render(&tokenizer, &messages, true).unwrap();
    let prompt_ids = tokenizer.encode(&rendered).unwrap();
    let tape = forced_tape(&[1]);
    let sampling = LlamaSampling::GumbelMax {
        temperature: 1e6,
        uniforms: &tape,
    };
    let direct = LlamaGenerator::new(&model, &tokenizer)
        .generate_ids(&prompt_ids, 1, sampling)
        .unwrap();
    let native = model
        .generate_chat_native(&tokenizer, template, &messages, 1, sampling)
        .unwrap();
    assert_eq!(native.prompt_ids(), prompt_ids);
    assert_eq!(native.generated_ids(), direct.generated_ids());
    assert_eq!(native.decoded(), direct.decoded());
    assert!(native.stopped());
}

#[test]
fn staged_native_generation_failure_rolls_back_single_and_batch_caches() {
    let (model, tokenizer, _) = make_model(8);
    let tape = forced_tape(&[3]);
    let sampling = LlamaSampling::GumbelMax {
        temperature: 1e6,
        uniforms: &tape,
    };
    let mut single = LlamaNativeGenerator::new(&model, &tokenizer);
    single.generate_text("a", 1, sampling).unwrap();
    let before = single.cache_len();
    single.inject_stage_failure(Some(1));
    assert!(matches!(
        single.generate_text("b", 1, sampling),
        Err(LlamaNativeGenerationError::Native(
            LlamaNativeError::InjectedStageFailure(1)
        ))
    ));
    assert_eq!(single.cache_len(), before);

    let mut batch = LlamaBatchNativeGenerator::new(&model, &tokenizer, 2).unwrap();
    batch
        .generate_texts(&["a", "bc"], 1, LlamaBatchSampling::Greedy)
        .unwrap();
    let before = batch.cache_lengths().to_vec();
    batch.inject_stage_failure(Some(1));
    assert!(matches!(
        batch.generate_texts(&["a", "bc"], 1, LlamaBatchSampling::Greedy),
        Err(LlamaNativeGenerationError::Native(
            LlamaNativeError::InjectedStageFailure(1)
        ))
    ));
    assert_eq!(batch.cache_lengths(), before);
}
