//! Hugging Face Hub access shared by the model loaders.

use std::path::PathBuf;

use hf_hub::{HFClientSync, HFError, HFRepositorySync, RepoTypeModel};

/// A model repository pinned to one revision; files resolve from the local
/// Hub cache and are downloaded on first use.
pub(crate) struct HubRepo {
    repo: HFRepositorySync<RepoTypeModel>,
    revision: String,
}

impl HubRepo {
    pub(crate) fn open(model_id: &str, revision: &str) -> Result<Self, HFError> {
        let (owner, name) = hf_hub::split_id(model_id);
        Ok(Self { repo: HFClientSync::new()?.model(owner, name), revision: revision.to_owned() })
    }

    pub(crate) fn get(&self, filename: &str) -> Result<PathBuf, HFError> {
        self.repo.download_file().filename(filename).revision(self.revision.as_str()).send()
    }
}
