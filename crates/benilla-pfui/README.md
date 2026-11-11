# pfUI compatibility extension

This is an optional compatibility extension for **brues-code/pfUI**, including its optional
modules. **Full pfUI is not supported yet.** The native and browser entry crates expose the
`pfui` feature; ordinary builds retain the stock API. The extension includes a headless audit
of the complete addon manifest and leaves unsupported APIs absent.

The contract under investigation is:

- [pfUI source revision `40dd9dd6`](https://github.com/brues-code/pfUI/tree/40dd9dd6db1cb8d15d565ec5bdaf39e70175e611)
- [ClassicAPI reference revision `917b1564`](https://github.com/brues-code/ClassicAPI/tree/917b15640e2bdc2c8f24e0e0d4d520514d76ff52)

ClassicAPI is a DLL for the original Windows client. Benilla must implement the API behavior
in its own engine; loading the DLL cannot provide those functions to its Lua VM.

## Isolation and integration

Most compatibility code lives in this crate. The engine connections are in new
`benilla-ui/src/script/pfui_host/` and `benilla-app/src/pfui.rs` files, with small
`#[cfg(feature = "pfui")]` hooks in the existing event, loader, item-cache, social-feed and
VM-creation paths. Attribute-bound frame hovers use a separate gated targeting module.
Auras use separate feeds beside the stock snapshots; cast and quest packets are observed through
additional handlers in the extension. Macro and reputation adapters live in new files beside
their stock modules and reuse their existing mutation and visible-row mechanisms.
The additional item metadata uses separate per-VM state, without extending stock item structs.
The vendored Lua parser has three guarded branches; it reads a **per-VM
registry flag**, absent in ordinary clients. Building this crate as a workspace member does
not activate the extension on stock VMs. Tests create stock and extended VMs side by side.

The client installs the extension on every in-game VM, including reloads, before stock
FrameXML and third-party addons load. Addon-private tables and `(addonName, privateTable)`
arguments flow through TOC chunks, XML includes and load-on-demand dependencies. Stock
`this`, `event` and `argN` globals remain available alongside modern positional handler arguments.
Class, race, faction and spell metadata comes from the owner's localized DBCs.

Build the experimental native integration with:

```sh
cargo build -p benilla --features pfui
# For a matching Turtle installation instead of Vanilla:
cargo build -p benilla --features pfui-turtle
```

Place the separately obtained pfUI checkout under `benilla-config/AddOns/pfUI` (or
`$BENILLA_HOME/AddOns/pfUI`), outside the WoW install. The features select
`pfUI_ClassicAPI.toc` or `pfUI_Turtle.toc`, respectively. Browser builds can select the same
features on `wenilla` and need their normal wasm toolchain. The `webgpu,pfui-turtle`
combination passes a wasm compile check; browser runtime behavior remains unverified.

To run the native Turtle build, point it at the matching data and the state folder containing
`AddOns/pfUI`:

```sh
WOW_DATA='/path/to/Turtle_WoW/Data' WOW_HOST=localhost \
BENILLA_HOME='/path/to/benilla-config' \
cargo run -p benilla --features pfui-turtle
```

Log in normally and select a character. For Vanilla data, use `--features pfui` instead.
Keep the feature on the run command: an ordinary `cargo run -p benilla` builds the stock client.
Use `/pfui` in game to open its configuration. This is an experimental native setup; full
gameplay support and browser support remain unverified.

The ordinary `pfUI.toc` deliberately loads only `API_Check.lua`. ClassicAPI selects the full
flavor TOC in the original client. The full TOC still checks `CLASSIC_API_VERSION >= 11510`,
and without it raises a dialog at every login that says pfUI is disabled and takes the keyboard
focus. The extension therefore publishes `CLASSIC_API_VERSION = 11510`, the minimum pfUI asks
for. Deviation: this is the pfUI-used surface, not ClassicAPI's complete API, so another addon
that tests the version can find a function missing. `BenillaClassicAPICompatibility.emulated`
identifies the implementation and `fullClassicAPI` stays false. The extension separately implements
pfUI's Nampower integration and returns `3, 0, 0` from `GetNampowerVersion`, which enables those
pfUI modules. `BenillaNampowerCompatibility.emulated` explicitly identifies the implementation;
`fullNampower` remains false. Installing Windows DLLs does not supply APIs to Benilla.

Benilla's [contribution rules](../../docs/CONTRIBUTING.md) exclude new client-mod APIs and
extension-only engine hooks. Consequently, compatibility APIs belong in this extension/fork.
An independently demonstrated 1.12 bug fix can be submitted upstream separately. This is a
portability boundary, not a promise that upstream will merge the parser carry. The extension
builds against this checkout's benilla revision; upstream provides no stable extension API.

## Available primitives

| Surface | Behavior |
| --- | --- |
| Lua expressions | Opt-in `...`, `#`, `%`; Lua string methods; `string.match`/`gmatch`/`reverse`; `select`; missing math functions |
| Stock Lua behavior | Legacy `arg` tables and WoW's positional `string.format` stay usable; no filesystem or OS libraries are opened |
| Addon file calls | Shared private tables across TOC chunks, XML inclusions and demand-loaded addons; enclosing contexts are restored |
| Widgets and events | `SetSize`/`GetSize`, `SetShown`, `SetFormattedText`, `HookScript`, `hooksecurefunc`, `RegisterUnitEvent`, `IsEventRegistered`; numeric-string paint colours, modern arguments with legacy globals |
| Frame attributes | Case-insensitive storage, modifier lookups, recursive-safe `OnAttributeChanged`, target/assist/focus/menu/stop-cast actions, GUID-directed spells and named/inline macros through the ordered stock call stream; frame unit attributes publish real mouseover picks |
| `C_Timer` | `After`, `NewTimer`, `NewTicker`, cancellation, bounded iterations, next-frame dispatch; one anonymous frame, idle with no timers |
| `RunNextFrame` | A one-shot timer on the following UI frame |
| General helpers | `Mixin`, `CreateFromMixins`, `tInvert`, `SafePack`, `Round`, `Clamp`, `CopyTable`, object/texture pools |
| Creature metadata | Host DBC class names and `C_CreatureInfo.GetRaceInfo` / `GetFactionInfo` |
| Spells | Localized DBC metadata, legacy ten-value tuples with player/pet book and name/rank lookup, live power costs, cooldowns, usability and GUID-based range queries through the existing engine validators |
| Caster stats | Per-school damage modifiers from the player's descriptor and haste percentages from each visible unit's cast-speed multiplier |
| Known spells | Aura-effect searches over the complete known-spell set, including hidden talents, rather than only visible spellbook rows |
| Player states | Descriptor form IDs and mount state, real stealth auras, authorized mount/form cancellation from DBC effects; warrior stances remain active |
| Nampower integration | Explicit 3.0.0 compatibility identity, packet-backed spell/aura/heal events, raw aura slot and stack edges, event CVars, GUID-directed named casts, forced/no-queue spellbook overloads, GCD/non-GCD/on-swing queues, ground quickcasting, bounded server-rejection retries, prioritized queued scripts and pfUI queue indicator |
| Observed casts | Cast/channel queries over spell packets, remote channels confirmed by live descriptors, stable per-cast identifiers, pushback and modern start/success/stop/failure events |
| Auras | Seven pfUI-used queries, raw descriptor slots, pagination, fill buffers, polarity/source/crowd-control filters, DBC-backed overflow classification, refresh notifications and authorized player-aura cancellation |
| Units and clock | Stock, focus, GUID and nameplate query tokens; local-player GUID checks, typed/raw power, NPC subtitles, party/raid membership and realm-local calendar |
| Focus | `FocusUnit`/`ClearFocus`, target-chain reads over visible-unit snapshots, real GUID targeting and focus identity/health/name/level/target events |
| Nameplates | Real pooled frames, stable unit-token slots, target chains and frame-valued creation, unit-added and readable unit-removed events |
| New items | Bag glow flags keyed by instance GUID, login baseline, move-stable flags, removal/clear operations and change-only notifications |
| Creature identity | Live NPC/pet template IDs, creature GUID fallback outside view, and no pet-number/entry confusion |
| Items | Cached 18-value item info, classification, qualities, levels, family, stack/sell limits, usability, equipped checks and async item helpers |
| Bags | Item IDs, charge counts, openable/instance-lock checks, paced atomic moves and swaps through the existing sender; counts include cold bank objects |
| Actions and equipment | `GetActionInfo`, `GetInventoryItemID` over stock inventory/inspect snapshots, player durability and equipment-slot constants |
| Social records | Structured friend and `/who` records, localized class tokens, online counts, real friend GUIDs, per-character notes and rate-limited silent name queries |
| Player cache | Opt-in `RememberPlayer`/`GetPlayerInfoByName`, real name-query replies, GUID and name collision handling, realm isolation and saved-variable persistence |
| Quests | Visible log IDs and cached server templates, including required items; accepted/removed events from real descriptor slots, successful turn-in events from server confirmations |
| Reputation | Structured visible rows and watched factions, retaining real IDs, folds, thresholds and synthetic-header exclusions |
| Macros | String-icon creation/editing using the existing macro table, caps, save mechanism and change events |
| Equipment sets | Per-character GUID snapshots, ignored slots, counts/locations, create/save/rename/delete, persistence, sequential atomic server swaps, draggable action buttons and summary tooltips with retained missing-item names |
| Inventory events | One `BAG_UPDATE_DELAYED` after stock bag handlers; `PLAYER_EQUIPMENT_CHANGED` compares actual instance GUIDs and publishes 1-or-nil presence |
| Cursor queries | `GetCursorInfo` reads the stock item, spell, macro, merchant or money payload and equipment sets; `CURSOR_CHANGED` records real payload transitions |
| Tooltip subjects | Item, spell and unit queries follow native builders, cleared callbacks, async spell views and hide; focus/GUID unit tooltips reuse the stock builder; `GameTooltip_Hide` closes the real tooltip for friend-note callbacks |
| Loot history | Bounded real roll history with player votes, observed dice, winner/pass completion, late name-query identities and modern notifications |
| Icons | MPQ and loose icon catalogs, filtered item/spell pools, spellbook and equipment extras |
| Map reveal | Current viewed zone and all its DBC overlays, actual BLP tile dimensions, partial edges and supported HD tile layouts |
| Chat bubbles | Pooled real frames over the native lifetime, distance, visibility and projection decisions, with source regions pfUI can style |
| Merchants | Actual buyback count/item-ID helpers and paced grey-item sales over current carried GUIDs, canceled on vendor changes |
| Addon registry | Existence and optional dependencies over the real loader registry |
| Colors | The pfUI-used `ColorMixin` methods, `CreateColor`, `C_ColorUtil.GenerateTextColorCode` |
| Callbacks | `CallbackRegistryMixin` and `EventRegistry` owner registration/removal, real frame-event callbacks, dispatch and reentrant subscriptions |
| `EventUtil` | `ContinueOnPlayerLogin`, `ContinueOnAddOnLoaded`; once per registration, immediate when already ready |
| `C_EncodingUtil` | Eight profile-sharing calls: CBOR/JSON serialization, zlib compression, Base64 encoding, and their inverses |
| ClassicAPI globals | `IsModifierKeyDown`, `strtrim`, item-ID companions of the stock link queries, `FindSpellBookSlotByID`, `GetMacroSpell`, `GetFactionInfoByID`, `UnitNameFromGUID`, `GetPlayerInfoByGUID`, `GetUnitSpeed`, `GetInventoryItemRepairCost`, inbox and send-mail item links, `CreateSimpleTextureMarkup`, `CLASS_SORT_ORDER`, `INTERFACE_VERSION` |
| Tracking | `GetNumTrackingTypes`, `GetTrackingInfo`, `SetTracking` over the player's book spells whose DBC effect applies a tracking aura |
| Equipment flyout | `GetInventoryItemsForSlot` over worn and carried instances, packed locations and their unpack helpers |
| State events | `UPDATE_SHAPESHIFT_FORM`, `UPDATE_INVENTORY_DURABILITY`, `FACTION_STANDING_CHANGED`, `HEARTHSTONE_BOUND`, `GLOBAL_MOUSE_DOWN`/`GLOBAL_MOUSE_UP` |
| Item comparison | `GameTooltip:SetHyperlinkCompareItem` through the stock worn-item compare walk, and `SetAnchorType` |
| Inline textures | `|T…|t` in font strings, tooltips and chat lines: the later clients' size law, pixel crops, offsets and tint, drawn from the client's own texture store (`benilla-app/src/pfui/inline_texture.rs`); `GetCoinTextureString` |
| Unit model panes | An addon's own `PlayerModel:SetUnit` pane, such as pfUI's 3D portraits, is drawn by the client itself since benilla b727e30f: every character model frame draws its own `SetUnit` subject |
| Swing clock | `PLAYER_SWING(swingDuration, swingType)` rebuilt from the player's white hits, on-next-swing abilities, swing-resetting casts, auto-repeat shots and parried attacks; at most one event per weapon per frame (`src/swing.rs`) |
| Colours and labels | Colour methods on the stock `*_FONT_COLOR` tables, `WHITE_FONT_COLOR`, and English labels for the equipment manager, icon picker and friend notes |

Frame spell actions reuse the native cast binder and validators with the clicked unit's GUID,
without changing selection or falling back to self-cast. Ground spells use the unit's position.
Named and inline macros temporarily select the clicked unit, run through an addon `RunMacro`
when present or the stock chat-line event otherwise, and restore the prior selection. Lua target
queries inside the macro use the committed selection, and unwind even on runner errors. An absent
clicked unit sends nothing. Named item actions and `C_Item.UseItemByName` resolve a carried
instance by ID, link, decorated name, GUID or explicit location. They retain the unit GUID and
re-resolve the item's slot before sending, including after bind confirmation. Missing targets,
pending inventory operations, transformed items and confirmations from an older VM refuse the
request. Explicit missing or unknown unit strings refuse rather than using another target;
the unknown-string refusal differs from ClassicAPI's documented no-target treatment.
Direct item uses bypass vendor, bank and auto-equip bag-click modes. Attribute values such as
`"0 1"` and deprecated `bag`/`slot` attributes retain ClassicAPI's documented stock bag-click
path. Item queues and cursor/unit placement helpers are outside this subset.
The player cache defaults
to disabled; explicit `RememberPlayer` calls and real name-query replies populate it when enabled.
Replies assign wire gender directly; getters return the 2/3 sex convention pfUI's race icons use.
The optional visible-object sweep remains absent. Notes and cache records use `BenillaPfuiDB` in the
client's own saved-variable file, outside the WoW install. Note keys include both realm and
character, while player-cache keys include the realm.
The focus GUID is retained in session-owned memory across a UI reload, including a focus change
immediately before `/reload`. The replacement VM announces the restored focus after its live
unit snapshot and interface are ready. Logout and disconnect clear that memory, so another character or session
does not inherit it. Focus is not written to saved variables.

`GetShapeshiftFormID` returns the descriptor's DBC form ID, including `0` outside a form;
it does not return the stance bar's position. `IsMounted` reads the mount display descriptor.
Deviation: `IsStealthed` scans helpful stealth auras and excludes a mounted player, rather than
using the DLL's visibility bit shared by mounts and some forms. Mount and form cancellation
match actual DBC aura effects and pass the stock cancellation authorization gate; warrior
stance cancellation is explicitly ignored. No spell-name or form-ID lists are hardcoded.

Atomic bag operations use a bounded queue and wait for both descriptor endpoints after each
server move. Whole-stack moves use the normal swap/merge packet: this server rejects splitting
an entire source stack. Counted moves reject a mismatched or overflowing destination before
sending, and a rejected or unanswered operation cancels the remaining batch. Deviation:
`BAG_UPDATE_DELAYED` waits until the extension's batch settles, so pfUI's placement phase reads
all consolidation results. Stock `BAG_UPDATE` events remain immediate. The stock cursor is
untouched; these APIs return acceptance into the queue, rather than network-send success.
Native probes exercise pfUI sorting after a real stack split, preserve item totals, and restore
the original slots and counts. Bank and specialized-bag sorting still need live coverage.

The aura feed retains actual descriptor slots, player durations when known, and unknown sources
as unknown. Successful observed casts supply aura ownership and DBC duration for their hit
targets; the player's server duration remains authoritative. Ownership survives unchanged
snapshots and clears on removal or slot reuse. Source tokens include group pets, focus and
nameplates. Newly learned ownership and refreshed player duration packets notify `UNIT_AURA`
after updating the modern snapshot; unchanged timers produce no repeated notifications.
Roster auras remain available when a group member or pet leaves object range.

`CROWD_CONTROL` and its negation use DBC mechanics and active control/slow aura effects.
The modern projection identifies overflow debuffs through `AURA_IS_DEBUFF` and intrinsically
harmful/control effects, keeping ordinary harmful slots first in filtered queries. Cancelable
buffs and negative-range slots retain server classification. The descriptor carries no general
polarity field, so unknown or ambiguous custom spells still use their slot range; complete Turtle
overflow parity needs live comparison. Auras already present at login retain unknown ownership,
as their descriptors omit the caster. Cast observation supports
ordinary visible unit tokens, local channels and observed remote channels. Remote channel timing
starts with an observed `SMSG_SPELL_START`; `UNIT_CHANNEL_SPELL` confirms the channel and stops it
when the descriptor clears, changes spell or leaves view. A cast-time prelude waits for
`SMSG_SPELL_GO` before becoming a channel. Start, success and stop retain the cast identifier
across target, focus and nameplate aliases. Remote durations use the caster-level-scaled DBC;
remote talent modifiers and pushback are unavailable, and a channel first encountered midway
has no invented start time. Targeting/sent/quiet events, remaining focus/nameplate event coverage
and spell-interruption immunity still need integration. Interruption
immunity is returned as `nil` when unknown. Query availability alone does not establish full
Nampower behavior.

Nampower compatibility follows the addon-facing contracts in
[EVENTS.md](https://github.com/brues-code/nampower/blob/2c7f51cd436e9b0ce88271e84ce5a77c19f2c23b/EVENTS.md)
and [SCRIPTS.md](https://github.com/brues-code/nampower/blob/2c7f51cd436e9b0ce88271e84ce5a77c19f2c23b/SCRIPTS.md).
The optional extension owns its `NP_` settings in `BenillaPfuiDB`, independently of Benilla's
stock CVar table. The five pfUI event switches default to zero and pfUI enables them at login.
The ordinary spell queue defaults to a 500 ms window and 55 ms buffer, binds the target at
press time, replaces an earlier queued GCD spell, and re-enters the normal validator at release.
`QueueSpellByName` forces a request outside the window and retains it through a long cast.
`CastSpellByNameNoQueue` and `CastSpellNoQueue` bypass queuing for that individual request;
they still use the normal validator. Spellbook casts accept slots, learned names and
`spellId:123` strings, including pet spells and explicit unit targets. Self-directed casts
do not depend on an unrelated selection remaining streamed.
Explicit `target` tokens and their target chains resolve after preceding macro selections have
committed, then retain that GUID while queued. A missing explicit target never falls back to
the current selection or to self-casting.
Up to six pending non-GCD spells retain FIFO order and take priority over a queued GCD spell.
Nampower's pinned README specifies six entries, while its `QueueSpellByName` script reference
says five; this implementation follows the README pending an original-DLL capacity comparison.
A repeated spell replaces its own entry; replacing a shared nonzero start-recovery category is
opt-in through `NP_ReplaceMatchingNonGcdCategory`. Successful non-GCD sends apply the
`NP_NonGcdBufferTimeMs` spacing (100 ms by default). `SpellStopCasting` clears all spell queues
before release; session exit drops them. Auto-attack, item and profession actions continue through their
existing paths. Unknown `NP_` settings delegate to
stock handling; their existence is not advertised. `GetCastInfo` uses the same epoch as
`GetTime` and reads the actual GCD leg; `GetPlayerAuraDuration` uses raw slots and server timers.
`QueueScript` retains one replaceable script inside the ordinary queue window, with priority
1 before spells, 2 between the non-GCD and GCD lanes, or 3 after both. Idle requests and requests
outside the window execute immediately through the ordered call stream. Queued scripts wait
for the actual GCD even without a cast bar, read selection when they execute, and clear on
session exit or VM replacement. Invalid priorities raise instead of retaining an unexecutable
request. Scripts do not generate spell-queue icon events.
Terrain spells use `NP_QueueTargetingSpells` (default 1) and
`NP_TargetingQueueWindowMs` (500 ms). Queued terrain casts retain the request's finite world hit
and quickcast through the ordinary validator and terrain commit, regardless of the ordinary
quickcast setting. Explicit unit targets retain their GUID and use that unit's position at
release. `NP_QuickcastTargetingSpells` and `NP_QuickcastOnDoubleCast` default to zero; enabling
them also affects individual no-queue calls. A sky aim invents no coordinate. Stop-targeting
and world right presses cancel queued terrain requests; a held cursor payload pre-empts the
right-press cancellation. Terrain quickcasts emit cast types 4/5. Cursor movement timing still
needs comparison with the original DLL; ordinary manual terrain-click event coverage is pending.
The independent on-swing lane defaults off (`NP_QueueOnSwingSpells`). When enabled, it retains
one replaceable request until the server clears the existing melee spell and the cast validator
is ready, without estimating swing timers. `NP_OnSwingBufferCooldownMs` defaults to 500 ms and
prevents an immediate duplicate queue after a committed strike. Queue events use codes 0/1;
priority-3 scripts wait for this lane too. Strike combat behavior still needs a controlled
GM-off comparison; the command-stream tests do not establish that live behavior.
`NP_RetryServerRejectedSpells` defaults to one. Only our own successfully dispatched spell
requests can retry, and only after server reasons `ITEM_NOT_READY` (0x28), `NOT_READY` (0x3c)
or `SPELL_IN_PROGRESS` (0x61). Retries retain the resolved unit GUID or terrain wire coordinates,
pass through the stock validator and queue spacing, and yield to newer queued player requests.
Local refusals and explicit no-queue casts never retry. Acceptance cancels a waiting retry
without discarding ordinary player queues; stop-casting, terrain cancellation, session exit and
VM replacement discard the relevant retry state. Disabling the CVar also discards waiting retries.
Deviation: retries are limited to three resends within two seconds of the first send; the public
contract does not specify these bounds. Item/GO casts and ordinary manual terrain-click sends
remain outside this retry subset. Synthetic decoded-verdict tests verify this behavior through
the registered stock handlers and real outgoing commands; live transient rejection and original-DLL
retry timing still need comparison. The 1.12 verdict has no cast identifier, so same-spell
overlapping verdicts cannot be distinguished beyond the engine's single outstanding-cast slot.
The added functionality stays in extension files, with small feature-gated stock hooks.
Extension snapshots and event delivery participate in the stock `UiFeed` phase. Cast snapshots
and spell runtime data precede Nampower cast events; spell runtime data follows the stock unit,
spell-modifier and action feeds. Aura edges follow the visible-aura projection. Buffered events
wait for the in-game UI; logout/disconnect cleanup does not wait for it.

Equipment sets distinguish duplicate item IDs by their complete 64-bit GUID strings. The executor
re-resolves a GUID after each server update, so paired rings/trinkets need only one atomic swap.
Missing items and bank items are skipped, and empty saved slots do not unequip current items.
The equip executor leaves the cursor untouched. Deviation from ClassicAPI: `EQUIPMENT_SWAP_FINISHED` waits for observed
placement, rather than reporting dispatch success immediately; a rejected, deferred or unanswered
swap fails after ten seconds. Native probes exercise restoring real clothing, equipment-set
button dragging, rename/icon updates and persistence across reload. Bind-confirm, combat,
disconnection and insufficient-bag-space cases remain unverified.

Encoding uses standard CBOR, JSON, zlib and RFC 4648 Base64 rather than an executable Lua
serializer. Cycles, unsupported values and malformed input raise Lua errors. Profile payloads
are limited to 8 MiB and nesting to 64 levels. Nonempty contiguous integer-key tables become
arrays; mixed keys remain CBOR maps; empty tables become objects. JSON object keys must be
strings. These are the supported profile-data rules, not a claim of complete ClassicAPI
serialization parity. Live exchange with a DLL-generated pfUI profile still needs verification.
The native synthetic profile fixture round-trips nested tables, arrays, booleans, fractional
numbers, empty tables and UTF-8 text through the CBOR/zlib/Base64 pipeline. The current
original-client comparison is blocked before its codec probe: the isolated Wine client with
ClassicAPI and Nampower authenticates to localhost, then raises error 132 at `0x00477D15`
during character selection. That failed reference run is not evidence of codec compatibility
or incompatibility, and no player profile was imported or changed.
A separate login-screen probe confirms that `C_EncodingUtil` is absent from the original
client's glue VM, so the comparison requires entering the world.

Quest lifecycle feeds ignore login and reload bulk sync, wait for cached accepted rows, and
distinguish confirmed turn-ins from abandons. Deviation: a collapsed quest has no stock public
row index, so its accepted event carries index 0 and its real questID. A complete-to-incomplete
transition produces the modern accepted event; the stock client's chat announcement remains on
its existing new-quest/cleared-failure path.

Spell-damage bonuses currently use the server's player modifier fields. Servers that omit
computed gear or talent contributions need an additional client calculation; these reads do
not claim ClassicAPI's complete computed-value behavior on every realm. Haste uses the actual
unit cast-speed multiplier and returns zero when the field or unit is unavailable.

Equipment action bindings persist with their character's sets. They share the engine cursor
and grid, displace real actions through the ordinary server clear queue, and yield to server
actions changed by another client. A local cursor tag is rejected by the stock packet packer;
no new server action type or generated macro is used. Renames, icon updates and deletion reach
every bound button. Locked items and active casts prevent an action-button equipment swap.

The globals outside the `C_` namespaces were found by a live inventory: every global, upper-case
constant and widget method pfUI's source names was looked up in the running VM, and every event
it registers was compared with the stock event list and the events this client fires. The static
audit below only sees dotted names. The item-ID companions read the entry out of the link the
stock builder returns for the same slot. `GetMacroSpell` resolves a `#showtooltip`/`#show`
argument or the first `/cast`, `/castsequence` or `CastSpellByName` through the player's books;
macro conditionals are not evaluated, so such a line resolves to nothing. `GetTrackingInfo`'s
`active` compares the spell's icon with the stock tracking texture. `GetFactionInfoByID` answers
a faction the player has a reputation slot for, with the documented neutral values before it is
met; a faction without a slot answers nil, where ClassicAPI reads `Faction.dbc`.
`GetPlayerInfoByGUID` and `UnitNameFromGUID` read a visible unit, the friends list, then this
session's name-query replies, and the saved cache only when it is enabled.
`GetInventoryItemsForSlot` also offers relics for the ranged slot, where they equip in 1.12.
`CLASS_SORT_ORDER` follows `ChrClasses.dbc` ID order. The equipment-manager, icon-picker and
friend-note labels are English in every locale unless the install's own strings define them.
`CreateSimpleTextureMarkup` returns the modern `|T…|t` escape. The client draws it only while
the extension is enabled; a stock client and the stock markup decoder still treat it as text.
The behaviour follows what ClassicAPI documents for its own backport: width, height and wrap
count the icon; an icon takes the string's alpha and its optional `r:g:b` tint, never the
text colour, and casts no shadow; one taller than the font grows its line and centres in it;
an outlined font keeps its ink clear of the icon; a link is hoverable across a tall icon
inside it; an EditBox shows the markup raw; and a line another player typed loses its
escapes before any addon sees it. Two deviations remain. A ScrollingMessageFrame row is a
whole font height here, so a tall icon in chat takes the next whole row rather than its exact
overflow. Atlas markup (`|A…|a`) is not implemented, since there is no atlas registry.
`GetCoinTextureString` crops the stock 64x16 money sheet; its three format strings are
English-neutral and a locale may define them first.

`PLAYER_SWING` reports the weapon's current speed from the unit fields at each reset. Two of
its rules are the server's and reach the client in no packet, so they follow vmangos: a cast
restarts the melee timers when the server announced its start and its `InterruptFlags` carry
the auto-attack bit, and a parry takes 40% of the weapon speed off the nearer swing, never
below 20%. `START_AUTOATTACK` and `STOP_AUTOATTACK`, which pfUI also registers, are not
fired: neither ClassicAPI nor Nampower documents them. pfUI's bar then hides when a swing
runs out with no new one. The swing clock is covered by tests and a check of the three
`Spell.dbc` columns against the Turtle install; it has not been watched in a live fight.

`GetCVar("gxResolution")` answers the window's size in device pixels. The client keeps that CVar
for the windowed size only and draws fullscreen at the monitor's own, while pfUI computes the
size of one device pixel from the string; with the stored default its borders came out twice
as thick on a high-resolution display.

The client's game menu has two rungs of its own, Edit Mode and Support. When pfUI's game-menu
skin has run they take the same button skin, so the menu reads as one.
The client also shrinks its own menu to 0.78 at every show; a skinned menu is put back to
UIParent's scale, where pfUI's one-pixel borders land on whole pixels.
Its rungs are stacked flush, where the stock menu leaves one unit between two; the extension
puts that unit back, or a highlighted rung's border is covered by its neighbour's backdrop.

The state events compare the stock snapshots after their feeds, with a baseline per VM and
character: a login, a reload and a relisted reputation table announce nothing.
`UPDATE_INVENTORY_DURABILITY` fires once when any worn slot's item or durability changes.
`HEARTHSTONE_BOUND` follows the server's `SMSG_PLAYERBOUND`. `PLAYER_SWING` is not fired, so
pfUI's swing timer stays idle. `SetHyperlinkCompareItem` returns 1 or nil as the stock compare
bindings do, never 0, and draws no per-stat change lines. `SetAnchorType` applies an offset for
the six owner-relative modes and leaves the recorded mode unchanged.

Timers use the engine's `GetTime` clock and require its ordinary `UiScript::tick` calls.
Callbacks scheduled during a timer callback wait until the next frame. A long frame fires a
ticker once rather than running an unbounded catch-up loop. Timer callback errors are caught,
as ClassicAPI documents; other addon-load and session errors remain visible.
Compatibility validation retains Lua's original `error` function because pfUI replaces the
global with a chat printer; an invalid callback must still raise before it can be installed.

## Audit the complete addon

Obtain a separate source checkout outside the game install:

```sh
git clone https://github.com/brues-code/pfUI.git /tmp/pfUI
git -C /tmp/pfUI checkout 40dd9dd6db1cb8d15d565ec5bdaf39e70175e611
cargo run -p benilla-pfui --features audit --bin pfui-audit -- /tmp/pfUI > /tmp/pfui-audit.json
```

To additionally load stock FrameXML and probe the first runtime blockers:

```sh
cargo run -p benilla-pfui --features audit --bin pfui-audit -- \
  /tmp/pfUI --data '/path/to/your/WoW/Data' > /tmp/pfui-audit.json
```

By default the tool reads the complete `pfUI_ClassicAPI.toc` and recursively follows XML file
inclusions. Add `--flavor turtle` to select the full Turtle manifest. Its JSON report contains
the source revision, file/compiler errors for both VMs, and each
dotted `C_*`/`EventUtil` reference with its source files and function availability. The scanner
excludes comments and strings. Static references can include optional branches and miss dynamic
lookups; they are a dependency inventory, not verified API contracts. Globals and widget
methods require additional runtime probes.

With `--data`, the audit uses the owner's MPQ chain directly. It builds stock FrameXML with
the app-owned camera CVars needed by its dropdown OnLoads and an explicit 1024×768 virtual
viewport. A dirty stock load prevents the addon bootstrap, so setup errors are never attributed
to pfUI. It has no renderer, network
session, character state or saved-variable persistence, and cannot verify real inventory moves,
rendering or gameplay. `stock_ui_attempted` and `bootstrap_attempted` distinguish skipped stages
from clean ones; `gameplay_verified` is always false. The command exits nonzero on unresolved
API references or load failures. It never installs addons, edits manifests, logs in, or writes
into the game install.

## Engine work required for full pfUI

| Area | Backing behavior needed |
| --- | --- |
| Units and focus | Broader focus/plate aura and descriptor event feeds, further query/verb token integration |
| Auras | Live verification of ambiguous/custom-spell Turtle overflow and broader aura-change coverage; pre-existing aura ownership is unavailable from the 1.12 descriptor, while observed successful casts provide caster/duration information |
| Casting and spells | Complete modern cast-event behavior and interruption classification; remote channel timing has no talent/pushback feed; runtime spell queries inherit existing engine validator limits |
| Remaining Nampower behavior | Item queues, bulk disenchanting and the broader Nampower API remain outside this subset; original-DLL retry timing, non-GCD capacity, terrain cursor timing and manual-click event coverage need comparison; the auto-attack component count is unavailable from the stock decoder; non-GCD/on-swing ordering is covered by command-capture tests, with combat behavior still unverified |
| Items and bags | Remaining instance helpers; broader multi-bag, bank and specialized-bag sorting verification |
| Equipment sets | Broader character/action-slot conflict verification; failures and bind-confirm behavior |
| Profile sharing | Live interoperability with the original ClassicAPI client; standard codec fixtures are covered |
| Browser | Live verification in a WebGPU browser; the Turtle feature passes the wasm compile check, while current live coverage is native |
| Remaining extras | Optional visible-player sweep, live vendor/group/totem checks and broader map/bubble validation |
| Widgets and globals | Further token-aware verbs, equipment IDs for uninspected players; broader click-casting and addon macro-runner live coverage |
| Swing timer | A live GM-off combat check of `PLAYER_SWING`; the off-hand delay at attack start, the reset after a weapon swap, and `C_SwingTimer` range checks |
| Item comparison | Per-stat change lines, which need item stat totals including on-equip spell bonuses |

The target is the complete pfUI manifest, including equipment management, sorting and sharing.
Implement only the functions/events that inventory and runtime probes demonstrate pfUI uses,
with real backing state. A namespace with empty functions would make the addon appear to load
while silently breaking features. Loading, drawing and using the UI need separate verification,
followed by reload/persistence checks and live tests on a dedicated account. All 108 dotted
API references discovered in the pinned manifest are available. That inventory excludes
widget methods, globals, event coverage and incomplete API behavior; complete inventory
coverage alone cannot prove full support.

## Checks

```sh
cargo test -p benilla-pfui --all-features
cargo fmt -p benilla-pfui -- --check
cargo clippy -p benilla-pfui --all-features --all-targets --no-deps -- -D warnings
cargo check -p benilla-app --features pfui-turtle
# With the wasm target and WASI SDK installed (scripts/web-setup.sh):
WASI_SDK=/path/to/wasi-sdk cargo check --target wasm32-unknown-unknown \
  -p wenilla --no-default-features --features webgpu,pfui-turtle
```

Tests cover VM isolation, legacy and modern varargs, stock formatting, timer cancellation and
reentrancy, event readiness, pfUI-style callbacks, standard transport fixtures, malformed imports,
source scanning, XML/nested addon contexts, widget inheritance, numeric-string paint channels,
unit-event filtering, item-cache
arrivals, cold-bank counting and bag operation queues. Further checks cover action/equipment
queries, attribute-driven selection, recursive attribute handlers, social records, note persistence,
player-cache name collisions, realm isolation and `/who` rate limiting. The audit deliberately
reports unresolved engine APIs whatever version the extension publishes.
Additional tests cover real overlay tile layouts and independent view tables, tooltip subject
clearing and late spell views, bubble replacement/recycling, loot result ordering, runtime spell
queries, quest bulk-sync suppression and turn-in/removal ordering, bag-event coalescing and
same-entry equipment-instance swaps. Additional tests cover nil aura slots, raw slot stability, filters/cancellation, cast identifiers,
typed mana/rage/energy, hidden known-spell queries, live/fallback creature IDs, reputation folds,
quest requirements, string-icon macros, plate token stability/removal/recycling, group rosters and GUID-based new-item tracking. Equipment tests retain duplicate GUIDs across saves,
resolve a paired swap with one send, wait for acknowledgements, fail on timeouts and reload saved
sets with character/realm isolation.
Aura regressions cover crowd-control/negated filters, overflow ordering and active-effect masks,
observed ownership during callbacks, timer refreshes without repeated notifications, slot reuse,
group-pet source tokens and roster fallback after a member streams out. Registered-packet channel
tests cover target/focus/nameplate queries and events, cast preludes, replacement, late failures,
unrelated instants, despawn and session cleanup. The shared aura clock has independent regressions
for uneven frames, hitches, refreshed durations and standalone feeds.
Native live probes verify self-cast Renew ownership and refresh, stable expiration with no
repeated idle aura notifications, and a six-second NPC First Aid channel on both target and
focus. Channel callbacks see the active snapshot at start and a cleared snapshot at stop;
pfUI renders both bars. The probe restores its character and event settings and reports no
Lua errors. These checks do not establish remote talent/pushback timing or browser rendering.
Equipment-button tests cover drag displacement, passive-spell refusal, rename/icon/delete
updates, cursor transition payloads, saved bindings, server action conflicts and locked-item
refusals through the actual swap queue.
Bag-worker tests cover chained partial/whole merges, split descriptor arrivals, placement over
confirmed slots, all-or-nothing counts, timeout and failure cancellation. Name-reply tests cover
opt-in writes, corrected wire gender, recycled names, realm isolation and late loot identities
without duplicate completion events.
Click-casting tests cover GUID snapshots, named/inline macro bodies, scoped Lua target reads,
selection restoration on errors, missing units, form cancellation and warrior stance retention.
Caster tests cover the complete legacy metadata tuple, both spellbooks, passive and exact-rank
lookups, school indexing, haste/slow conversion and invalid or removed unit descriptors.
Item-action tests cover references and exact locations, ordered target chains, moved GUIDs,
vendor/bank bypass, locked and missing items, one-shot bind confirmation, transformed-item
refusal, VM replacement and replacing an extension confirmation with an ordinary bag click.
Native live item probes use the dedicated warrior's existing Hearthstone through four paths:
ID, exact bag location, named frame action and bag-slot frame action. Each receives one server
cast start, is cancelled before completion, ends idle and preserves the inventory snapshot.
The event settings are restored. These self-directed probes do not establish targeted scroll,
bind-confirmation or combat behavior; those paths currently have command-capture coverage.
Queue tests drive the real ordered-call applier and capture its client commands: forced casts
outside the window, same-macro double casts, no-queue refusals, long waits, replacement,
non-GCD priority/spacing/category settings, cancellation, VM replacement and self-casts with a
vanished unrelated selection. Further checks cover explicit target chains in macro order,
retained GUIDs after selection changes, all three queued-script priorities, replacement,
reentrancy and real GCD waits without a cast bar. The headless client includes the optional plugin when enabled;
source audits exclude its modules when the feature is disabled.
Terrain checks cover the targeting window and switch, source/destination packet coordinates,
retained hits after cursor/selection changes, quickcast settings, no-queue validation, sky aims
and cancellation before a ready release. On-swing checks cover default-off behavior, the
post-send duplicate window, long server-held strikes, replacement and normal-lane isolation.
Focus checks replace actual Lua VMs sharing the session memory, verify a single restore event
after the live snapshot, and drive the logout/disconnect handlers to check character isolation.
Native live coverage includes focus GUID/name retention and a visible pfUI focus frame after a
real reload, plus a queued self-directed Flare after a heal: one push/pop, one terrain cast and
one server confirmation without a cast failure. Temporary spellbook/settings changes are restored.
The rebuilt retry-enabled native client also passes the five self-heal queue scenarios: forced
double cast, explicit no-queue, preserving a forced request across a refused no-queue attempt,
cancellation and spell-ID overload. Queue icons finish hidden and the temporary queue-window
setting is restored. These self-directed checks do not establish hostile or GM-off combat behavior.
Auto-attack checks keep the ordinary attack toggle working while a cast is pending, without
creating a spell-queue entry.
Retry regressions dispatch decoded verdicts through the registered stock handlers and capture
outgoing commands. They cover all three transient reasons, terminal and unowned failures,
no-queue and CVar controls, resolved GUID and terrain retention, missing units, self-cast fallback,
minimum spacing, the resend/lifetime limits, late acceptance, newer user requests and session/VM
cleanup. Fresh scripts and pop-event callbacks can cancel a released request before it reaches
the sender; ordinary queued casts restore the selection present when they execute.

Global and event regressions cover the modifier rollup, literal trimming, item-ID companions,
spellbook slot lookup in both books, macro spell resolution and its refusals, tracking lists and
the cast a selection sends, met, unmet and absent factions, GUID name lookups, packed locations,
slot candidates by inventory type and hand, repair costs, mail links, unit speed, colour methods
on stock tables, form, durability and reputation edges with their baselines, the confirmed bind
and raw mouse buttons, and the worn-item comparison with re-anchoring.
Native live probes on the dedicated test character verify the new globals against real state
and the equipment flyout opening; one form event per stance change with the new form readable;
a +50 and -50 reputation change with the documented payload; one durability event per unequip
and re-equip; no event during login or a cross-map teleport; `GetUnitSpeed` reading 0, 7.0 while
walking and 0 again; and pfUI's shift-hover comparison placed beside the docked tooltip. A sweep
that opened pfUI's windows and ran all 3,446 hover handlers reports no pfUI error. The character
was restored after each probe. The hearthstone bind has handler coverage only.
On Turtle data the audit's stock-interface phase currently reports three errors in Turtle's own
FrameXML (the transmog, shop and Everlook frames), so it skips the pfUI bootstrap there; the
native client is the working check for that data.

All implementation in this crate is independently written under MIT OR Apache-2.0.
ClassicAPI's GPL implementation and the player's FrameXML are not bundled here.
