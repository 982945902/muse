use serde::{Deserialize, de::DeserializeOwned};
use std::{collections::HashMap, fs, sync::OnceLock};

const FIXTURE_ASSET_MANIFEST_PATH: &str = "fixtures/assets/manifest.json";

#[derive(Clone, Debug, Deserialize)]
pub struct FixtureAssetRef {
    pub asset_id: String,
    #[serde(default)]
    pub file_name: Option<String>,
    #[serde(default)]
    pub content_type: Option<String>,
}

#[derive(Clone, Debug)]
pub struct ResolvedFixtureAsset {
    pub asset_id: String,
    pub asset_path: String,
    pub file_name: String,
    pub content_type: String,
}

#[derive(Clone, Debug, Deserialize)]
struct FixtureAssetManifest {
    assets: Vec<FixtureAssetRecord>,
}

#[derive(Clone, Debug, Deserialize)]
struct FixtureAssetRecord {
    id: String,
    asset_path: String,
    file_name: String,
    content_type: String,
}

static FIXTURE_ASSET_INDEX: OnceLock<HashMap<String, FixtureAssetRecord>> = OnceLock::new();

pub fn load_json_fixture<T: DeserializeOwned>(path: &str) -> T {
    let raw = fs::read_to_string(path).unwrap_or_else(|error| {
        panic!("read json fixture `{path}`: {error}");
    });
    serde_json::from_str(&raw).unwrap_or_else(|error| {
        panic!("parse json fixture `{path}`: {error}");
    })
}

pub fn load_fixture_bytes(path: &str) -> Vec<u8> {
    fs::read(path).unwrap_or_else(|error| panic!("read fixture asset `{path}`: {error}"))
}

pub fn load_asset_bytes(asset_id: &str) -> Vec<u8> {
    let asset = resolve_fixture_asset(&FixtureAssetRef {
        asset_id: asset_id.to_string(),
        file_name: None,
        content_type: None,
    });
    load_fixture_bytes(&asset.asset_path)
}

pub fn resolve_fixture_asset(asset_ref: &FixtureAssetRef) -> ResolvedFixtureAsset {
    let index = FIXTURE_ASSET_INDEX.get_or_init(load_fixture_asset_index);
    let record = index.get(&asset_ref.asset_id).unwrap_or_else(|| {
        panic!(
            "fixture asset id `{}` was not found in manifest",
            asset_ref.asset_id
        )
    });

    ResolvedFixtureAsset {
        asset_id: record.id.clone(),
        asset_path: record.asset_path.clone(),
        file_name: asset_ref
            .file_name
            .clone()
            .unwrap_or_else(|| record.file_name.clone()),
        content_type: asset_ref
            .content_type
            .clone()
            .unwrap_or_else(|| record.content_type.clone()),
    }
}

fn load_fixture_asset_index() -> HashMap<String, FixtureAssetRecord> {
    let manifest: FixtureAssetManifest = load_json_fixture(FIXTURE_ASSET_MANIFEST_PATH);
    manifest
        .assets
        .into_iter()
        .map(|asset| (asset.id.clone(), asset))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolves_asset_from_manifest_defaults() {
        let resolved = resolve_fixture_asset(&FixtureAssetRef {
            asset_id: "image_upload_green".to_string(),
            file_name: None,
            content_type: None,
        });

        assert_eq!(resolved.asset_id, "image_upload_green");
        assert_eq!(
            resolved.asset_path,
            "fixtures/assets/images/image_upload_green.png"
        );
        assert_eq!(resolved.file_name, "poster.png");
        assert_eq!(resolved.content_type, "image/png");
    }

    #[test]
    fn resolves_asset_with_fixture_overrides() {
        let resolved = resolve_fixture_asset(&FixtureAssetRef {
            asset_id: "image_upload_green".to_string(),
            file_name: Some("broken.png".to_string()),
            content_type: Some("application/x-broken".to_string()),
        });

        assert_eq!(resolved.asset_id, "image_upload_green");
        assert_eq!(resolved.file_name, "broken.png");
        assert_eq!(resolved.content_type, "application/x-broken");
        assert_eq!(
            resolved.asset_path,
            "fixtures/assets/images/image_upload_green.png"
        );
    }

    #[test]
    fn loads_asset_bytes_from_manifest_lookup() {
        let bytes = load_asset_bytes("profile_upload_docx");
        assert!(!bytes.is_empty());
        assert!(bytes.starts_with(&[0x50, 0x4b, 0x03, 0x04]));
    }
}
