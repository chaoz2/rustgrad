use super::{
    LLAMA_SIMPLE_CHAT_TEMPLATE, LlamaChatMessage, LlamaChatRole, LlamaPromptWorkflow,
    LlamaPromptWorkflowError,
};

#[test]
fn prompt_workflow_binds_fixture_generates_deterministically_and_fails_closed() {
    let bytes =
        super::model_tests::serialized_model_with_template(32, Some(LLAMA_SIMPLE_CHAT_TEMPLATE));
    let workflow = LlamaPromptWorkflow::from_gguf_bytes(&bytes).unwrap();
    let plain = workflow.generate("a", 2).unwrap();
    let repeated = workflow.generate("a", 2).unwrap();
    assert_eq!(plain, repeated);
    assert_eq!(plain.generation().prompt_ids(), &[3]);
    assert!(plain.generation().generated_ids().len() <= 2);

    let chat = workflow
        .generate_chat(
            &[LlamaChatMessage::new(LlamaChatRole::User, "a").unwrap()],
            1,
        )
        .unwrap();
    assert!(chat.rendered_prompt().contains("assistant"));
    assert!(chat.generation().generated_ids().len() <= 1);

    let before = workflow.generate("a", 1).unwrap();
    assert!(workflow.generate("a", 32).is_err());
    assert_eq!(before, workflow.generate("a", 1).unwrap());
    assert!(matches!(
        LlamaPromptWorkflow::from_gguf_bytes(b"not a gguf"),
        Err(LlamaPromptWorkflowError::Gguf(_))
    ));
    let unsupported = super::model_tests::serialized_model_with_template(32, Some("{{ bad }}"));
    assert!(matches!(
        LlamaPromptWorkflow::from_gguf_bytes(&unsupported),
        Err(LlamaPromptWorkflowError::Chat(_))
    ));
}
