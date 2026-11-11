local bridge = ...
local frame = CreateFrame('Frame')
local queue, context = {}, nil
local function stop()
  queue, context = {}, nil
  frame:SetScript('OnUpdate', nil)
end
frame:RegisterEvent('MERCHANT_SHOW')
frame:RegisterEvent('MERCHANT_CLOSED')
frame:SetScript('OnEvent', stop)
local function update()
  if context ~= bridge.context() then stop(); return end
  local guid = table.remove(queue, 1)
  if guid then bridge.sell(guid) end
  if #queue == 0 then stop() end
end
function C_MerchantFrame.SellAllJunkItems()
  queue, context = bridge.snapshot(), bridge.context()
  if not context or #queue == 0 then stop(); return end
  frame:SetScript('OnUpdate', update)
end
