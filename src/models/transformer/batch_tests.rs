use super::{
    LLAMA_SIMPLE_CHAT_TEMPLATE, LlamaBatchCache, LlamaBatchGenerationError, LlamaBatchGenerator,
    LlamaBatchSampling, LlamaChatError, LlamaChatMessage, LlamaChatRole, LlamaChatTemplate,
    LlamaGenerator, LlamaModel, LlamaModelError, LlamaSampling,
    model_tests::{
        VOCAB, assert_close, make_model, serialized_model_with_numeric_template,
        serialized_model_with_template,
    },
};
use crate::{
    TensorData,
    tokenizer::{SimpleTokenizer, TokenizerConfig, TokenizerPreset},
};

#[test]
fn padded_batch_matches_independent_full_sequences() {
    let (model, _, _) = make_model(8);
    let rows = vec![vec![3, 4, 5], vec![6, 3]];
    let plan = model.plan_batch(&rows).unwrap();
    assert_eq!(
        plan.graph().shape(plan.logits_node()).unwrap().dims(),
        &[2, 3, VOCAB]
    );
    let batched = plan.execute().unwrap();
    for (actual, tokens) in batched.iter().zip(&rows) {
        assert_close(actual, &model.forward(tokens).unwrap(), 4e-5);
    }
}

#[test]
fn batch_cache_matches_full_for_tokens_and_uneven_chunks() {
    let (model, _, _) = make_model(8);
    let rows = [vec![3, 4, 5], vec![6, 3, 4]];
    let full = model.forward_batch(&rows).unwrap();

    let mut token_cache = LlamaBatchCache::new(model.config().clone(), 2).unwrap();
    let mut collected = [Vec::new(), Vec::new()];
    for position in 0..3 {
        let step = token_cache
            .forward(&model, &[vec![rows[0][position]], vec![rows[1][position]]])
            .unwrap();
        for row in 0..2 {
            collected[row].extend_from_slice(step[row].values());
        }
    }
    assert_eq!(token_cache.lengths(), &[3, 3]);
    for row in 0..2 {
        assert_close(
            &TensorData::new([3, VOCAB], collected[row].clone()).unwrap(),
            &full[row],
            4e-5,
        );
    }

    let mut chunk_cache = LlamaBatchCache::new(model.config().clone(), 2).unwrap();
    let first = chunk_cache.forward(&model, &[vec![3, 4], vec![6]]).unwrap();
    let second = chunk_cache.forward(&model, &[vec![5], vec![3, 4]]).unwrap();
    assert_eq!(chunk_cache.lengths(), &[3, 3]);
    for row in 0..2 {
        let mut values = first[row].values().to_vec();
        values.extend_from_slice(second[row].values());
        assert_close(
            &TensorData::new([3, VOCAB], values).unwrap(),
            &full[row],
            4e-5,
        );
    }
}

#[test]
fn batch_cache_rejects_invalid_rows_without_partial_commit() {
    let (model, _, _) = make_model(4);
    let mut cache = LlamaBatchCache::new(model.config().clone(), 2).unwrap();
    let mut control = LlamaBatchCache::new(model.config().clone(), 2).unwrap();
    cache.forward(&model, &[vec![3, 4], vec![5]]).unwrap();
    control.forward(&model, &[vec![3, 4], vec![5]]).unwrap();
    let before = cache.lengths().to_vec();
    assert_eq!(
        cache
            .forward(&model, &[vec![5], vec![6, 3, 4, 5]])
            .unwrap_err(),
        LlamaModelError::BatchContextLength {
            row: 1,
            requested: 5,
            maximum: 4
        }
    );
    assert_eq!(cache.lengths(), before);
    let actual = cache.forward(&model, &[vec![5], vec![6]]).unwrap();
    let expected = control.forward(&model, &[vec![5], vec![6]]).unwrap();
    for row in 0..2 {
        assert_close(&actual[row], &expected[row], 0.0);
    }
    let committed = cache.lengths().to_vec();
    assert_eq!(
        cache
            .forward(&model, &[vec![3], vec![VOCAB as u32]])
            .unwrap_err(),
        LlamaModelError::BatchTokenOutOfRange {
            row: 1,
            token: VOCAB as u32,
            vocab_size: VOCAB,
        }
    );
    assert_eq!(cache.lengths(), committed);
    assert_eq!(
        cache
            .forward(&model, &[Vec::new(), Vec::new()])
            .unwrap_err(),
        LlamaModelError::EmptyBatchStep
    );
    assert_eq!(cache.lengths(), committed);
}

#[test]
fn batch_greedy_generation_matches_independent_generators() {
    let (model, tokenizer, _) = make_model(8);
    let mut batch = LlamaBatchGenerator::new(&model, &tokenizer, 2).unwrap();
    let actual = batch
        .generate_texts(&["a", "bc"], 2, LlamaBatchSampling::Greedy)
        .unwrap();
    for (row, prompt) in ["a", "bc"].iter().enumerate() {
        let expected = LlamaGenerator::new(&model, &tokenizer)
            .generate_text(prompt, 2, LlamaSampling::Greedy)
            .unwrap();
        assert_eq!(actual.sequences()[row].prompt_ids(), expected.prompt_ids());
        assert_eq!(
            actual.sequences()[row].generated_ids(),
            expected.generated_ids()
        );
        assert_eq!(actual.sequences()[row].decoded(), expected.decoded());
        assert_eq!(actual.sequences()[row].stopped(), expected.stopped());
    }
}

#[test]
fn batch_gumbel_tape_is_step_batch_vocab_row_major_with_independent_stops() {
    let (model, tokenizer, _) = make_model(8);
    let mut tape = vec![1e-9; 2 * 2 * VOCAB];
    tape[1] = 0.9;
    tape[VOCAB + 3] = 0.9;
    tape[3 * VOCAB + 2] = 0.9;
    let mut generator = LlamaBatchGenerator::new(&model, &tokenizer, 2).unwrap();
    let generated = generator
        .generate_texts(
            &["a", "b"],
            2,
            LlamaBatchSampling::GumbelMax {
                temperature: 1e6,
                uniforms: &tape,
            },
        )
        .unwrap();
    assert_eq!(generated.sequences()[0].generated_ids(), &[1]);
    assert_eq!(generated.sequences()[1].generated_ids(), &[3, 2]);
    assert!(generated.sequences().iter().all(|row| row.stopped()));
    assert_eq!(generator.cache_lengths(), &[1, 2]);

    let before = generator.cache_lengths().to_vec();
    assert_eq!(
        generator
            .generate_texts(
                &["a", "b"],
                2,
                LlamaBatchSampling::GumbelMax {
                    temperature: 1.0,
                    uniforms: &[0.5]
                },
            )
            .unwrap_err(),
        LlamaBatchGenerationError::UniformTapeLength {
            required: 48,
            actual: 1
        }
    );
    assert_eq!(generator.cache_lengths(), before);
    assert_eq!(
        generator
            .generate_texts(&["abcd", "a"], 5, LlamaBatchSampling::Greedy)
            .unwrap_err(),
        LlamaBatchGenerationError::ContextLength {
            row: 0,
            requested: 9,
            maximum: 8
        }
    );
    assert_eq!(generator.cache_lengths(), before);
}

#[test]
fn checked_llama_chat_template_matches_fallback_and_rejects_other_jinja() {
    let bytes = serialized_model_with_template(16, Some(LLAMA_SIMPLE_CHAT_TEMPLATE));
    let file = crate::gguf::read_gguf(&bytes).unwrap();
    let (_, tokenizer) = LlamaModel::from_gguf(&file).unwrap();
    let template = LlamaChatTemplate::from_gguf(&file).unwrap();
    assert!(template.metadata_present());
    let messages = [
        LlamaChatMessage::new(LlamaChatRole::System, "a").unwrap(),
        LlamaChatMessage::new(LlamaChatRole::User, "b").unwrap(),
    ];
    assert_eq!(
        template.render(&tokenizer, &messages, true).unwrap(),
        "<|start_header_id|>system<|end_header_id|>\n\na</s>\
<|start_header_id|>user<|end_header_id|>\n\nb</s>\
<|start_header_id|>assistant<|end_header_id|>\n\n"
    );

    // tinygrad's fallback routes all three Llama BPE labels through the same
    // header/end-turn branch; this is not a Qwen/OLMo/Kimi template alias.
    for preset in [TokenizerPreset::LlamaV3, TokenizerPreset::LlamaBpe] {
        let alternate = SimpleTokenizer::new(
            Vec::<(String, u32)>::new(),
            [("</s>".to_owned(), 1)],
            TokenizerConfig {
                preset,
                bos_id: None,
                eos_id: 1,
                eot_id: None,
            },
        )
        .unwrap();
        assert_eq!(
            template.render(
                &alternate,
                &[LlamaChatMessage::new(LlamaChatRole::User, "a").unwrap()],
                true,
            )
            .unwrap(),
            "<|start_header_id|>user<|end_header_id|>\n\na</s>\
<|start_header_id|>assistant<|end_header_id|>\n\n"
        );
    }
    let qwen = SimpleTokenizer::new(
        Vec::<(String, u32)>::new(),
        [("</s>".to_owned(), 1)],
        TokenizerConfig {
            preset: TokenizerPreset::Qwen2,
            bos_id: None,
            eos_id: 1,
            eot_id: None,
        },
    )
    .unwrap();
    assert_eq!(
        template
            .render(&qwen, &[LlamaChatMessage::new(LlamaChatRole::User, "a").unwrap()], true)
            .unwrap_err(),
        LlamaChatError::UnsupportedPreset(TokenizerPreset::Qwen2)
    );

    let bytes = serialized_model_with_template(
        16,
        Some("{% for message in messages %}{{ message }}{% endfor %}"),
    );
    let file = crate::gguf::read_gguf(&bytes).unwrap();
    assert_eq!(
        LlamaChatTemplate::from_gguf(&file).unwrap_err(),
        LlamaChatError::UnsupportedJinja
    );

    let bytes = serialized_model_with_numeric_template(16);
    let file = crate::gguf::read_gguf(&bytes).unwrap();
    assert!(matches!(
        LlamaChatTemplate::from_gguf(&file).unwrap_err(),
        LlamaChatError::Metadata(crate::gguf::GgufMetadataAccessError::TypeMismatch { .. })
    ));
}

#[test]
fn serialized_gguf_runs_reader_chat_tokenizer_model_and_batch_generation() {
    let bytes = serialized_model_with_template(16, Some(LLAMA_SIMPLE_CHAT_TEMPLATE));
    let file = crate::gguf::read_gguf(&bytes).unwrap();
    let (model, tokenizer) = LlamaModel::from_gguf(&file).unwrap();
    let chat = LlamaChatTemplate::from_gguf(&file).unwrap();
    let prompts = ["a", "b"].map(|content| {
        let rendered = chat
            .render(
                &tokenizer,
                &[LlamaChatMessage::new(LlamaChatRole::User, content).unwrap()],
                true,
            )
            .unwrap();
        tokenizer.encode(&rendered).unwrap()
    });
    assert_eq!(prompts[0], vec![7, 10, 8, 9, 3, 1, 7, 11, 8, 9]);
    let mut tape = vec![1e-9; 2 * VOCAB];
    tape[1] = 0.9;
    tape[VOCAB + 2] = 0.9;
    let mut generator = LlamaBatchGenerator::new(&model, &tokenizer, 2).unwrap();
    let output = generator
        .generate_ids(
            &prompts,
            1,
            LlamaBatchSampling::GumbelMax {
                temperature: 1e6,
                uniforms: &tape,
            },
        )
        .unwrap();
    assert_eq!(output.sequences()[0].generated_ids(), &[1]);
    assert_eq!(output.sequences()[0].decoded(), "</s>");
    assert_eq!(output.sequences()[1].generated_ids(), &[2]);
    assert_eq!(output.sequences()[1].decoded(), "<|im_end|>");
    assert!(output.sequences().iter().all(|row| row.stopped()));
}
