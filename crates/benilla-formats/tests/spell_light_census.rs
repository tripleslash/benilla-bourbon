//! Instrument: the light `synthesize_spell_light` derives for named spell models, side by side —
//! the missile against the kit effects (cast hands, burning auras) that sit against a body.

use benilla_formats::{open_chain, parse_m2_particle_emitters, spell_light};

/// `WOW_SPELL_MODELS="Spells\\A.m2;Spells\\B.m2"` overrides the default fire set.
#[test]
#[ignore = "instrument: run by hand — cargo test -p benilla-formats --test spell_light_census -- --ignored --nocapture"]
fn spell_light_census() {
    let data = benilla_formats::wow_data_or_skip!();
    let mut chain = open_chain(&data).expect("open vanilla patch chain");
    let paths = std::env::var("WOW_SPELL_MODELS").unwrap_or_else(|_| {
        [
            "Spells\\Fireball_Missile_High.m2",
            "Spells\\Fireball_Missile_Low.m2",
            "Spells\\Fire_Precast_Hand.m2",
            "Spells\\Fire_PreCast_Med_Hand.m2",
            "Spells\\Fire_PreCast_High_Hand.m2",
            "Spells\\Fire_Cast_Hand.m2",
            "Spells\\Immolate_State.m2",
            "Spells\\Immolate_State_Base.m2",
            "Spells\\Fire_DOT_State_Chest.m2",
            "Spells\\FireNova_State.m2",
        ]
        .join(";")
    });
    for path in paths.split(';') {
        let Ok(bytes) = chain.read_file(path) else {
            println!("{path:40} MISSING");
            continue;
        };
        let emitters = parse_m2_particle_emitters(&bytes).unwrap_or_default();
        match spell_light::synthesize_spell_light(path, emitters.iter()) {
            Some(s) => println!(
                "{path:40} {:8} intensity {:.2} colour {:?} emitter {} of {} onset {:.2}",
                s.bucket,
                s.intensity,
                s.color.map(|c| (c * 100.0).round() / 100.0),
                s.emitter,
                emitters.len(),
                s.onset,
            ),
            None => println!("{path:40} no light ({} emitters)", emitters.len()),
        }
    }
}
