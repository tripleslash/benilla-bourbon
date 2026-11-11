pfUI:RegisterSkin("Help", function ()
  local rawborder, border = GetBorderSize()
  local bpad = rawborder > 1 and border - GetPerfectPixel() or GetPerfectPixel()

  StripTextures(HelpFrame)
  CreateBackdrop(HelpFrame, nil, nil, .75)
  CreateBackdropShadow(HelpFrame)

  HelpFrame.backdrop:SetPoint("TOPLEFT", 4, -4)
  HelpFrame.backdrop:SetPoint("BOTTOMRIGHT", -44, 12)
  HelpFrame:SetHitRectInsets(4,44,4,12)
  EnableMovable(HelpFrame)

  SkinCloseButton(HelpFrameCloseButton, HelpFrame.backdrop, -6, -6)

  local title = GetNoNameObject(HelpFrame, "FontString", "ARTWORK", HELP_FRAME_TITLE)
  title:ClearAllPoints()
  title:SetPoint("TOP", HelpFrame.backdrop, "TOP", 0, -10)

  -- HelpFrameHome
  StripTextures(HelpFrameHomeIssues)
  SkinButton(HelpFrameHomeIssues)
  HelpFrameHomeIssues:SetScript("OnMouseUp", nop)
  HelpFrameHomeIssues:SetScript("OnMouseDown", nop)

  SkinButton(HelpFrameHomeCancel)

  -- HelpFrameGM
  SkinButton(HelpFrameGMCancel)

  SkinButton(HelpFrameGMBack)
  HelpFrameGMBack:ClearAllPoints()
  HelpFrameGMBack:SetPoint("RIGHT", HelpFrameGMCancel, "LEFT", -2*bpad, 0)

  -- HelpFrameOpenTicket
  StripTextures(HelpFrameOpenTicketDivider)

  CreateBackdrop(HelpFrameOpenTicketScrollFrame, nil, nil, .75)
  StripTextures(HelpFrameOpenTicketScrollFrame)
  SkinScrollbar(HelpFrameOpenTicketScrollFrameScrollBar)
  HelpFrameOpenTicketText:SetMaxLetters(5000)

  SkinButton(HelpFrameOpenTicketCancel)

  SkinButton(HelpFrameOpenTicketSubmit)
  HelpFrameOpenTicketSubmit:ClearAllPoints()
  HelpFrameOpenTicketSubmit:SetPoint("RIGHT", HelpFrameOpenTicketCancel, "LEFT", -2*bpad, 0)

  -- HelpFrameGeneral
  StripTextures(HelpFrameGeneralButton)
  SkinButton(HelpFrameGeneralButton)
  HelpFrameGeneralButton:SetScript("OnMouseUp", nop)
  HelpFrameGeneralButton:SetScript("OnMouseDown", nop)

  StripTextures(HelpFrameGeneralButton2)
  SkinButton(HelpFrameGeneralButton2)
  HelpFrameGeneralButton2:SetScript("OnMouseUp", nop)
  HelpFrameGeneralButton2:SetScript("OnMouseDown", nop)

  SkinButton(HelpFrameGeneralCancel)

  SkinButton(HelpFrameGeneralBack)
  HelpFrameGeneralBack:ClearAllPoints()
  HelpFrameGeneralBack:SetPoint("RIGHT", HelpFrameGeneralCancel, "LEFT", -2*bpad, 0)

  -- HelpFrameHarassment
  StripTextures(HelpFrameHarassmentDivider)
  StripTextures(HelpFrameHarassmentDivider2)

  StripTextures(HelpFrameVerbalHarassmentButton)
  SkinButton(HelpFrameVerbalHarassmentButton)
  HelpFrameVerbalHarassmentButton:SetScript("OnMouseUp", nop)
  HelpFrameVerbalHarassmentButton:SetScript("OnMouseDown", nop)

  StripTextures(HelpFramePhysicalHarassmentButton)
  SkinButton(HelpFramePhysicalHarassmentButton)
  HelpFramePhysicalHarassmentButton:SetScript("OnMouseUp", nop)
  HelpFramePhysicalHarassmentButton:SetScript("OnMouseDown", nop)

  SkinButton(HelpFrameHarassmentCancel)

  SkinButton(HelpFrameHarassmentBack)
  HelpFrameHarassmentBack:ClearAllPoints()
  HelpFrameHarassmentBack:SetPoint("RIGHT", HelpFrameHarassmentCancel, "LEFT", -2*bpad, 0)

  -- TicketStatusFrame
  CreateBackdrop(TicketStatusFrame, nil, true, .75)
  TicketStatusFrame:ClearAllPoints()
  TicketStatusFrame:SetPoint("TOP", 0, -5)
  UpdateMovable(TicketStatusFrame)
  function TicketStatusFrame_OnEvent()
    if ( event == "PLAYER_ENTERING_WORLD" ) then
      GetGMTicket()
    else
      if ( arg1 ~= 0 ) then
        this:Show()
        refreshTime = GMTICKET_CHECK_INTERVAL
      else
        this:Hide()
      end
    end
  end
end)
