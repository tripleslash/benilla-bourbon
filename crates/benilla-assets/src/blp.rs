//! BLP to Bevy [`Image`] loader, in the [`BlpVariant`] the load settings choose.

use bevy::asset::io::Reader;
use std::path::PathBuf;

use bevy::asset::{AssetLoader, AssetPath, LoadContext, RenderAssetUsages};
use bevy::image::{Image, ImageAddressMode, ImageFilterMode, ImageSampler, ImageSamplerDescriptor};
use bevy::reflect::TypePath;
use bevy::render::render_resource::{Extent3d, TextureDimension, TextureFormat};
use serde::{Deserialize, Serialize};

use benilla_formats::{blp_bytes_to_native_chain, blp_to_rgba, tga_texture_size, BlpMipChain};

use crate::gpu_blp::{for_upload, UploadChain};

/// Which on-GPU form a BLP decodes to.
#[derive(Default, Clone, Copy, PartialEq, Eq, Debug, Serialize, Deserialize)]
pub enum BlpVariant {
    /// World and model albedo, the default: non-sRGB, so shader math stays in gamma bytes; the
    /// authored mips as stored (the 1.12 client never regenerates them); the process filter policy.
    #[default]
    WorldArt,
    /// Emissive billboard (the sun and moon discs): `Rgba8UnormSrgb`, clamp, mip 0 only.
    Sprite,
    /// Gamma-lane effect sprite (rain splashes, weather mist): `Rgba8Unorm`, clamp, mip 0 only, as
    /// these quads are mostly magnified and their thin cut-out arms fade out down the mips.
    Effect,
    /// The snow flake: [`Self::Effect`] with the authored mips kept. The reference draws it as
    /// `GL_POINTS` with `GL_COORD_REPLACE` (`0x678610`; device init `0x59cf30`-`0x59cf58`), 14 px
    /// at the eye and 1 px past 46 yd, where mip 0 alone aliases into speckle.
    PointSprite,
    /// A minimap tile (ADT `map<X>_<Y>` or a WMO group's interior tile): [`map_tile_image`].
    MapTile,
    /// OS cursor image: `Rgba8UnormSrgb`, single mip.
    Cursor,
}

/// Per-load settings for [`BlpImageLoader`]: choose the [`BlpVariant`] via `load_with_settings`.
#[derive(Default, Clone, Serialize, Deserialize)]
pub struct BlpLoaderSettings {
    pub variant: BlpVariant,
}

/// Bevy [`AssetLoader`] decoding `*.blp` to [`Image`].
#[derive(Default, TypePath)]
pub struct BlpImageLoader;

impl AssetLoader for BlpImageLoader {
    type Asset = Image;
    type Settings = BlpLoaderSettings;
    type Error = std::io::Error;

    async fn load(
        &self,
        reader: &mut dyn Reader,
        settings: &Self::Settings,
        ctx: &mut LoadContext<'_>,
    ) -> Result<Image, Self::Error> {
        let mut bytes = Vec::new();
        reader.read_to_end(&mut bytes).await?;
        // Everything past here is CPU over bytes already in memory, with no `JsValue` and no
        // `LoadContext`, so it can leave the frame thread: `frame_budget::offload` runs it on a
        // worker where the build has one, and in the metered slot where it does not. It is taken
        // after the read, so the frame is never billed for the fetch. The decode's two inputs from
        // the loader are resolved first, because neither can cross: the settings' variant, and the
        // address mode riding the asset path (`crate::texture_url`: one `.blp` can be uploaded
        // under two samplers).
        let variant = settings.variant;
        let wrap = crate::sampler_mode_of(&ctx.path().to_string());
        // The reader hands over the first candidate of `TextureCreate`'s chain that opens
        // (`0x449d90`), which is a `.tga` where no `.blp` is there or the name asked for it.
        let tga_size = (!bytes.starts_with(b"BLP"))
            .then(|| tga_texture_size(&bytes))
            .flatten();
        // A model's texture keeps the levels the files beside it hold (`0x4493a0`). They are read
        // here, where the `LoadContext` is, and go along to the decode.
        let mut mip_files = Vec::new();
        if let (Some((w, h)), BlpVariant::WorldArt) = (tga_size, variant) {
            let stem = crate::tga_stem(&ctx.path().path().to_string_lossy());
            for n in 1..benilla_formats::full_mip_count(w, h) as u32 {
                let name = crate::tga_chain::mip_file_name(&stem, n);
                let path = AssetPath::from_path_buf(PathBuf::from(name))
                    .with_source(ctx.path().source().clone().into_owned());
                let Ok(file) = ctx.read_asset_bytes(path).await else {
                    break;
                };
                mip_files.push(Some(file));
            }
        }
        crate::frame_budget::offload(crate::frame_budget::Lane::Art, move || {
            if tga_size.is_none() {
                return decode(variant, wrap, &bytes);
            }
            let chain = crate::tga_chain::tga_texture_chain(&bytes, |n| {
                mip_files.get_mut(n as usize - 1).and_then(Option::take)
            })
            .ok_or_else(|| std::io::Error::other("TGA header makes no texture"))?;
            Ok(tga_image(chain, variant, wrap))
        })
        .await
    }

    fn extensions(&self) -> &[&str] {
        &["blp"]
    }
}

/// One BLP decoded into its [`Image`], the loader's whole CPU half. A free function, so a `move`
/// closure with a `Send + 'static` bound (`crate::frame_budget::offload`) can name it; that bound is
/// what keeps it off the page.
fn decode(variant: BlpVariant, wrap: (bool, bool), bytes: &[u8]) -> Result<Image, std::io::Error> {
    let to_io = |e: anyhow::Error| std::io::Error::other(format!("{e:#}"));
    Ok(match variant {
        BlpVariant::WorldArt => world_art_image(
            // The blocks go straight to the GPU where it can take them; `for_upload` decodes only
            // when it cannot, and hands back the matching format.
            for_upload(blp_bytes_to_native_chain(bytes).map_err(to_io)?),
            wrap,
        ),
        BlpVariant::Sprite => {
            let (w, h, rgba) = blp_to_rgba(bytes).map_err(to_io)?;
            sprite_image(w, h, rgba)
        }
        BlpVariant::MapTile => {
            let (w, h, rgba) = blp_to_rgba(bytes).map_err(to_io)?;
            map_tile_image(w, h, rgba)
        }
        BlpVariant::Effect => {
            let (w, h, rgba) = blp_to_rgba(bytes).map_err(to_io)?;
            effect_image(w, h, rgba)
        }
        BlpVariant::PointSprite => {
            point_sprite_image(for_upload(blp_bytes_to_native_chain(bytes).map_err(to_io)?))
        }
        BlpVariant::Cursor => {
            let (w, h, rgba) = blp_to_rgba(bytes).map_err(to_io)?;
            cursor_image(w, h, rgba)
        }
    })
}

/// A `.tga` candidate's chain (`0x449d90`) as the `variant` lane draws it: the lanes that draw
/// one level take level 0.
fn tga_image(chain: BlpMipChain, variant: BlpVariant, wrap: (bool, bool)) -> Image {
    let (w, h) = (chain.width, chain.height);
    match variant {
        BlpVariant::WorldArt => world_art_image(for_upload(chain), wrap),
        other => {
            let rgba = chain.mips[0].clone();
            match other {
                BlpVariant::Sprite => sprite_image(w, h, rgba),
                BlpVariant::MapTile => map_tile_image(w, h, rgba),
                BlpVariant::Effect => effect_image(w, h, rgba),
                BlpVariant::PointSprite => point_sprite_image(for_upload(BlpMipChain {
                    mips: vec![rgba],
                    ..chain
                })),
                _ => cursor_image(w, h, rgba),
            }
        }
    }
}

/// The gamma lanes' uncompressed format: non-sRGB, so the GPU does not linearize on sample.
const GAMMA_BYTES: TextureFormat = TextureFormat::Rgba8Unorm;

fn world_art_image(upload: UploadChain, wrap: (bool, bool)) -> Image {
    let UploadChain { chain, format } = upload;
    let levels = chain.mips.len() as u32;
    let mut data = Vec::with_capacity(chain.mips.iter().map(Vec::len).sum());
    for mip in &chain.mips {
        data.extend_from_slice(mip);
    }
    // `new_uninit`: `Image::new` would clone mip 0 only to feed a length assert meant for
    // uncompressed formats.
    let mut image = Image::new_uninit(
        Extent3d {
            width: chain.width,
            height: chain.height,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        format,
        // The render world takes the chain on extract rather than cloning it and keeping a
        // main-world copy (bevy_render `render_asset.rs`); nothing reads world art main-side.
        RenderAssetUsages::RENDER_WORLD,
    );
    image.data = Some(data);
    image.texture_descriptor.mip_level_count = levels;
    let mode = |repeat: bool| {
        if repeat {
            ImageAddressMode::Repeat
        } else {
            ImageAddressMode::ClampToEdge
        }
    };
    // The reference's terrain, WMO and M2 textures take the mip filter and anisotropy from the
    // process policy (`0x449ae0`, `crate::tex_filter`).
    let filter = crate::tex_filter::tex_filter();
    image.sampler = ImageSampler::Descriptor(ImageSamplerDescriptor {
        address_mode_u: mode(wrap.0),
        address_mode_v: mode(wrap.1),
        mag_filter: ImageFilterMode::Linear,
        min_filter: ImageFilterMode::Linear,
        mipmap_filter: filter.mipmap_filter(),
        anisotropy_clamp: filter.anisotropy_clamp(),
        ..Default::default()
    });
    image
}

/// Emissive billboard: sRGB, so an unlit pass returns the authored gamma bytes to screen.
fn sprite_image(width: u32, height: u32, rgba: Vec<u8>) -> Image {
    let mut image = Image::new(
        Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        rgba,
        TextureFormat::Rgba8UnormSrgb,
        RenderAssetUsages::RENDER_WORLD,
    );
    image.sampler = ImageSampler::Descriptor(ImageSamplerDescriptor {
        address_mode_u: ImageAddressMode::ClampToEdge,
        address_mode_v: ImageAddressMode::ClampToEdge,
        mag_filter: ImageFilterMode::Linear,
        min_filter: ImageFilterMode::Linear,
        ..Default::default()
    });
    image
}

/// A minimap tile, sampled as the reference samples every tile (the tile loader `0x6d9ed0`): clamp,
/// `GL_LINEAR` with no mips, anisotropy 1 and `GL_SKIP_DECODE_EXT`, so the hardware filters the
/// gamma bytes. An sRGB upload would darken every alpha edge against black, so the tile uploads as
/// [`GAMMA_BYTES`] and the shader converts after the filter.
fn map_tile_image(width: u32, height: u32, rgba: Vec<u8>) -> Image {
    effect_image(width, height, rgba)
}

fn effect_image(width: u32, height: u32, rgba: Vec<u8>) -> Image {
    let mut image = Image::new(
        Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        rgba,
        GAMMA_BYTES,
        RenderAssetUsages::RENDER_WORLD,
    );
    image.sampler = ImageSampler::Descriptor(ImageSamplerDescriptor {
        address_mode_u: ImageAddressMode::ClampToEdge,
        address_mode_v: ImageAddressMode::ClampToEdge,
        mag_filter: ImageFilterMode::Linear,
        min_filter: ImageFilterMode::Linear,
        ..Default::default()
    });
    image
}

/// No anisotropy: a point sprite is square and screen-aligned, with no anisotropic footprint.
fn point_sprite_image(upload: UploadChain) -> Image {
    let UploadChain { chain, format } = upload;
    let levels = chain.mips.len() as u32;
    let mut data = Vec::with_capacity(chain.mips.iter().map(Vec::len).sum());
    for mip in &chain.mips {
        data.extend_from_slice(mip);
    }
    let mut image = Image::new_uninit(
        Extent3d {
            width: chain.width,
            height: chain.height,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        format,
        RenderAssetUsages::RENDER_WORLD,
    );
    image.data = Some(data);
    image.texture_descriptor.mip_level_count = levels;
    image.sampler = ImageSampler::Descriptor(ImageSamplerDescriptor {
        address_mode_u: ImageAddressMode::ClampToEdge,
        address_mode_v: ImageAddressMode::ClampToEdge,
        mag_filter: ImageFilterMode::Linear,
        min_filter: ImageFilterMode::Linear,
        mipmap_filter: crate::tex_filter::tex_filter().mipmap_filter(),
        ..Default::default()
    });
    image
}

fn cursor_image(width: u32, height: u32, rgba: Vec<u8>) -> Image {
    Image::new(
        Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        rgba,
        TextureFormat::Rgba8UnormSrgb,
        RenderAssetUsages::default(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy::asset::io::memory::{Dir, MemoryAssetReader};
    use bevy::asset::io::AssetSourceBuilder;
    use bevy::asset::{AssetApp, AssetPlugin, AssetServer, Assets};
    use bevy::prelude::*;
    use std::path::Path;

    /// An uncompressed 32-bit TGA, B G R A per texel, `pixel` repeated.
    fn tga(side: u16, pixel: [u8; 4]) -> Vec<u8> {
        let mut b = vec![0u8; 18];
        b[2] = 2;
        b[12..14].copy_from_slice(&side.to_le_bytes());
        b[14..16].copy_from_slice(&side.to_le_bytes());
        b[16] = 32;
        b[17] = 8 | 0x20;
        for _ in 0..side as usize * side as usize {
            b.extend_from_slice(&pixel);
        }
        b
    }

    /// Load `url` through [`BlpImageLoader`] over a memory source named `mpq` holding `files`.
    fn load(files: &[(&str, Vec<u8>)], url: &str) -> Option<Image> {
        let dir = Dir::default();
        for (name, bytes) in files {
            dir.insert_asset(Path::new(name), bytes.clone());
        }
        let mut app = App::new();
        app.register_asset_source(
            crate::MPQ_SOURCE,
            AssetSourceBuilder::new(move || Box::new(MemoryAssetReader { root: dir.clone() })),
        );
        app.add_plugins((MinimalPlugins, AssetPlugin::default()));
        app.init_asset::<Image>();
        app.register_asset_loader(BlpImageLoader);
        let handle: Handle<Image> = app.world().resource::<AssetServer>().load(url.to_string());
        for _ in 0..2000 {
            app.update();
            if let Some(image) = app.world().resource::<Assets<Image>>().get(&handle) {
                return Some(image.clone());
            }
            std::thread::sleep(std::time::Duration::from_millis(2));
        }
        None
    }

    /// The loader's own content check takes a `.tga` candidate (no `BLP` magic, a TGA header), and
    /// builds its chain from the files beside it (`0x4493a0`) and a box filter below them.
    #[test]
    fn the_loader_takes_a_tga_candidate_and_its_mip_files() {
        let image = load(
            &[
                ("tex/foo@cc.blp", tga(16, [10, 20, 30, 255])),
                ("tex/foo_mip1.tga", tga(8, [200, 200, 200, 255])),
            ],
            "mpq://tex/foo@cc.blp",
        )
        .expect("a TGA behind a .blp name loads");
        assert_eq!((image.width(), image.height()), (16, 16));
        assert_eq!(image.texture_descriptor.mip_level_count, 5);
        let data = image.data.as_ref().unwrap();
        assert_eq!(data.len(), 1024 + 256 + 64 + 16 + 4);
        assert_eq!(&data[..4], &[30, 20, 10, 255], "level 0 from the file");
        assert!(
            data[1024..1280]
                .chunks(4)
                .all(|p| p == [200, 200, 200, 255]),
            "level 1 is the file's"
        );
        assert_eq!(
            &data[1360..],
            &[200, 200, 200, 255],
            "1x1 is filtered from level 1"
        );
    }

    /// A gated `.tga` the decoder refuses is opaque white at the gated size, not a load error
    /// (`0x44a2eb` into `0x448920`).
    #[test]
    fn a_gated_tga_that_will_not_decode_loads_as_white() {
        let mut bad = tga(8, [0; 4]);
        bad.truncate(18);
        let image =
            load(&[("tex/bad.blp", bad)], "mpq://tex/bad.blp").expect("white, not an error");
        assert_eq!((image.width(), image.height()), (8, 8));
        assert_eq!(image.texture_descriptor.mip_level_count, 4);
        assert!(image.data.as_ref().unwrap().iter().all(|&b| b == 0xff));
    }
}
