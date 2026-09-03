//! Explicit module traversal and graph-independent, versioned parameters.
//!
//! A [`Parameter`] owns only host state. [`Parameter::bind`] snapshots that state
//! into a graph-local input leaf, and [`Module::input_bindings`] retrieves the
//! values captured by that graph. Replacing a parameter never mutates an
//! existing graph or changes the values its leaves observe.

mod activation;
mod bert;
mod bert_model;
mod bert_pretraining;
mod bert_qa;
mod conv;
mod efficientnet;
mod embedding;
mod init;
mod linear;
mod log_softmax;
mod norm;
mod output;
mod parameter;
mod pool;
mod recurrent;
mod regularization;
mod resnet;
mod resnet_metal;
mod sequential;
mod shape;
mod softmax;
mod state;
mod transformer;

pub use activation::{ActivationFn, GELU, GeluApproximation, ReLU, SiLU, Sigmoid, Tanh};
pub use bert::{BertEncoderLayer, BertEncoderLayerConfig};
pub use bert_model::{BertEmbeddings, BertEncoder, BertModel, BertModelConfig};
pub use bert_pretraining::{
    BertForPretraining, BertLMPredictionHead, BertPooler, BertPreTrainingHeads,
    BertPredictionHeadTransform, BertPretrainingAccuracy, BertPretrainingOutput,
};
pub use bert_qa::BertForQuestionAnswering;
pub use conv::{Conv1d, Conv1dOptions, Conv2d, ConvTranspose1d, ConvTranspose2d};
pub use efficientnet::{EfficientNet, EfficientNetConfig, MBConvBlock, MBConvBlockConfig};
pub use embedding::Embedding;
pub use linear::Linear;
pub use log_softmax::LogSoftmax;
pub use norm::{
    BatchNorm, BatchNorm2d, BatchNorm3d, BatchNormOutput, GroupNorm, InstanceNorm, LayerNorm,
    LayerNorm2d, PendingBatchNormStats, RMSNorm,
};
pub use output::Argmax;
pub use parameter::{Parameter, ParameterId, ParameterSnapshot};
pub use pool::{
    AdaptiveAvgPool1d, AdaptiveAvgPool2d, AdaptiveMaxPool1d, AdaptiveMaxPool2d, AvgPool1d,
    AvgPool2d, MaxPool1d, MaxPool2d,
};
pub use recurrent::{LSTM, LSTMCell, LSTMOutput, LSTMState};
pub use regularization::{Dropout, ModeDropout};
pub use resnet::{
    BasicBlock, BasicBlockConfig, Bottleneck, BottleneckConfig, ResNet, ResNetBlock, ResNetConfig,
    ResNetDepth, ResNetForwardOutput, ResNetOutput,
};
pub use resnet_metal::{ResNetMetalError, ResNetMetalPlan, ResNetMetalRun, ResNetMetalSession};
pub use sequential::{ModeSequential, Sequential};
pub use shape::Flatten;
pub use softmax::Softmax;
pub use state::{
    CastPolicy, LiveStateDict, LoadReport, Mode, ModeForwardOutput, ModeModuleForward, Module,
    ModuleForward, PendingModeEffects, RealizedBatchNormStats, StateDict, StateKind,
    StrictStateLoadLimits, get_parameters, get_state_dict,
};
pub use transformer::TransformerBlock;

pub(crate) use parameter::{ParameterRestore, next_version, restore_parameters};
pub(crate) use state::module_input_node_bindings;

#[cfg(test)]
mod conv_tests;
#[cfg(test)]
mod layer_tests;
#[cfg(test)]
mod norm_tests;
#[cfg(test)]
mod pool_tests;
#[cfg(test)]
mod recurrent_tests;
#[cfg(test)]
mod state_tests;
