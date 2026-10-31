//! A sound file's own sample rate, off its header: what [`super::web_decode`] decodes at so the
//! browser does not resample it first. Nearly every WAV the client plays is 22050 Hz PCM.

/// The rate a WAV's `fmt ` chunk or an MP3's first frame header states; `None` for anything else.
pub(super) fn of(bytes: &[u8]) -> Option<u32> {
    wav_rate(bytes).or_else(|| mp3_rate(bytes))
}

fn wav_rate(bytes: &[u8]) -> Option<u32> {
    if bytes.get(0..4)? != b"RIFF" || bytes.get(8..12)? != b"WAVE" {
        return None;
    }
    let mut at = 12;
    loop {
        let id = bytes.get(at..at + 4)?;
        let size = u32::from_le_bytes(bytes.get(at + 4..at + 8)?.try_into().ok()?) as usize;
        if id == b"fmt " {
            let rate = bytes.get(at + 12..at + 16)?;
            return Some(u32::from_le_bytes(rate.try_into().ok()?));
        }
        // A chunk's body is padded to an even length.
        at = at
            .checked_add(8)?
            .checked_add(size)?
            .checked_add(size & 1)?;
    }
}

fn mp3_rate(bytes: &[u8]) -> Option<u32> {
    let start = id3_size(bytes);
    bytes.get(start..)?.windows(4).find_map(|h| {
        if h[0] != 0xff || h[1] & 0xe0 != 0xe0 {
            return None;
        }
        let base = match (h[2] >> 2) & 3 {
            0 => 44_100,
            1 => 48_000,
            2 => 32_000,
            _ => return None,
        };
        let (version, layer) = ((h[1] >> 3) & 3, (h[1] >> 1) & 3);
        if layer == 0 || (h[2] >> 4) == 0xf {
            return None;
        }
        match version {
            3 => Some(base),
            2 => Some(base / 2),
            0 => Some(base / 4),
            _ => None,
        }
    })
}

/// The bytes an ID3v2 tag takes at the head of the file, 0 with none.
fn id3_size(bytes: &[u8]) -> usize {
    match bytes.get(0..10) {
        Some(h) if &h[..3] == b"ID3" => {
            10 + h[6..10]
                .iter()
                .fold(0usize, |size, b| (size << 7) | usize::from(b & 0x7f))
        }
        _ => 0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn wav(extra_chunk: &[u8], rate: u32) -> Vec<u8> {
        let mut out = b"RIFF\0\0\0\0WAVE".to_vec();
        if !extra_chunk.is_empty() {
            out.extend(b"LIST");
            out.extend((extra_chunk.len() as u32).to_le_bytes());
            out.extend(extra_chunk);
            out.resize(out.len() + extra_chunk.len() % 2, 0);
        }
        out.extend(b"fmt \x10\0\0\0\x01\0\x01\0");
        out.extend(rate.to_le_bytes());
        out.extend([0; 8]);
        out
    }

    #[test]
    fn a_wav_states_its_rate_after_any_chunk() {
        assert_eq!(of(&wav(&[], 22_050)), Some(22_050));
        assert_eq!(of(&wav(b"abc", 44_100)), Some(44_100), "an odd chunk pads");
    }

    #[test]
    fn an_mp3_states_its_rate_past_a_tag_and_by_version() {
        assert_eq!(of(&[0xff, 0xfb, 0x90, 0x00]), Some(44_100), "MPEG-1");
        assert_eq!(of(&[0xff, 0xf3, 0x90, 0x00]), Some(22_050), "MPEG-2");
        assert_eq!(of(&[0xff, 0xe3, 0x90, 0x00]), Some(11_025), "MPEG-2.5");
        let mut tagged = b"ID3\x03\0\0\0\0\0\x02\xff\xfb".to_vec();
        tagged.extend([0xff, 0xfb, 0x94, 0x00]);
        assert_eq!(of(&tagged), Some(48_000), "the tag's bytes are not a frame");
    }

    #[test]
    fn anything_else_states_nothing() {
        assert_eq!(of(b""), None);
        assert_eq!(of(b"RIFF\0\0\0\0WAVEdata\xff\xff\xff\xff"), None);
        assert_eq!(of(&[0xff, 0xfb, 0x9c, 0x00]), None, "a reserved rate");
    }
}
