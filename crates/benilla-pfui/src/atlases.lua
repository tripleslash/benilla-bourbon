-- The atlases the ClassicAPI addon's templates name, each bound to the 1.12 art that draws the same
-- piece, at the rectangle stock FrameXML draws it with: the scroll track from
-- UIPanelScrollFrameTemplate's UI-Character-ScrollBar (FriendsFrame.xml, 64x256), the stepper
-- highlights as UIPanelScrollBarButton crops them, and the dropdowns from UIDropDownMenuTemplate's
-- label frame. 1.12 has no horizontal scroll art, so the horizontal pieces reuse the vertical.
local register = C_Texture.RegisterAtlas
local track = "Interface\\PaperDollInfoFrame\\UI-Character-ScrollBar"
local up = "Interface\\Buttons\\UI-ScrollBar-ScrollUpButton-Highlight"
local down = "Interface\\Buttons\\UI-ScrollBar-ScrollDownButton-Highlight"
local label = "Interface\\Glues\\CharacterCreate\\CharacterCreate-LabelFrame"

-- The track: the top cap and centre from the left column, the bottom cap from the right one's end.
register("UI-ClassicScrollBar-EndCap-Top", track, 31, 30, 0, 0.484375, 0, 0.1171875)
register("UI-ClassicScrollBar-Center", track, 31, 226, 0, 0.484375, 0.1171875, 1, false, true)
register("UI-ClassicScrollBar-EndCap-Bottom", track, 31, 30, 0.515625, 1, 0.296875, 0.4140625)
register("UI-ScrollBar-EndCap-Left", track, 30, 31, 0, 0.484375, 0, 0.1171875)
register("_UI-ScrollBar-Center", track, 226, 31, 0, 0.484375, 0.1171875, 1, true, false)
register("UI-ScrollBar-EndCap-Right", track, 30, 31, 0.515625, 1, 0.296875, 0.4140625)

register("UI-ScrollBar-ClassicScrollUpButton-Highlight", up, 16, 16, 0.25, 0.75, 0.25, 0.75)
register("UI-ScrollBar-ClassicScrollDownButton-Highlight", down, 16, 16, 0.25, 0.75, 0.25, 0.75)
register("UI-ScrollBar-ScrollLeftButton-Highlight", up, 16, 16, 0.25, 0.75, 0.25, 0.75)
register("UI-ScrollBar-ScrollRightButton-Highlight", down, 16, 16, 0.25, 0.75, 0.25, 0.75)

-- The dropdown's face: the label frame's middle, which every state shows.
for _, state in ipairs({"", "-disabled", "-hover-1", "-hover-2", "-open", "-pressed-1",
  "-pressed-2", "-pressedhover-1", "-pressedhover-2"}) do
  register("common-dropdown-c-button" .. state, label, 115, 64, 0.1953125, 0.8046875, 0, 1)
end
register("common-dropdown-c-bg", "Interface\\Tooltips\\UI-Tooltip-Background", 64, 64, 0, 1, 0, 1, true, true)
