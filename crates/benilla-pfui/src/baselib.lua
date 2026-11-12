-- ClassicAPI 1.20's base-library additions over Lua 5.0, as its API reference specifies them:
-- the error height an error handler reads, `scrub`, `difftime`, `table.removemulti`, a byte range
-- from `string.byte` and table replacements in `string.gsub`.

-- The handler runs at the raise, inside the failing script (the engine's error function), so the
-- height at its entry, less itself and the error function, is the failing code's. The addon's
-- function stays what geterrorhandler returns.
local nativeSet, nativeGet = seterrorhandler, geterrorhandler
local chosen, setHeight, errorHeight, inHandler
local function dispatch(message)
  local outer, outerIn = errorHeight, inHandler
  errorHeight, inHandler = GetCallstackHeight() - 2, true
  local ok, result = pcall(chosen, message)
  errorHeight, inHandler = outer, outerIn
  if not ok then error(result, 0) end
  return result
end
function seterrorhandler(handler)
  if type(handler) ~= "function" then return nativeSet(handler) end
  chosen = handler
  return nativeSet(dispatch)
end
function geterrorhandler()
  local held = nativeGet()
  if held == dispatch then return chosen end
  return held
end
function GetErrorCallstackHeight() return errorHeight or setHeight end
function SetErrorCallstackHeight(height)
  if height ~= nil and type(height) ~= "number" then error("Usage: SetErrorCallstackHeight([height])", 2) end
  setHeight = height
end
-- The locals of a level need the VM's own frames, which this client does not read: inside a
-- handler the text is empty, at other times there is none, as ClassicAPI's outside a handler.
function debuglocals()
  if inHandler then return "" end
end
-- A coroutine's stack is not read here: its text is empty. Otherwise the native's, one level
-- further out, so this wrapper is not a level of its own.
local nativeStack = debugstack
function debugstack(first, ...)
  if type(first) == "thread" then return "" end
  if first == nil then return nativeStack(2) end
  if type(first) ~= "number" then return nativeStack(2, first, unpack(arg)) end
  return nativeStack(first + 1, unpack(arg))
end

function scrub(...)
  local out = {}
  for i = 1, arg.n do
    local t = type(arg[i])
    if t == "boolean" or t == "number" or t == "string" then out[i] = arg[i] end
  end
  return unpack(out, 1, arg.n)
end

function difftime(time2, time1)
  if type(time2) ~= "number" then error("bad argument #1 to `difftime' (number expected)", 2) end
  return time2 - (tonumber(time1) or 0)
end

function table.removemulti(t, pos, count)
  local n = table.getn(t)
  if n == 0 then return end
  pos, count = pos or n, count or 1
  if pos < 1 or count < 0 or pos + count - 1 > n then error("parameters out of bounds", 2) end
  local removed = {}
  for i = 1, count do removed[i] = t[pos + i - 1] end
  for i = pos, n - count do t[i] = t[i + count] end
  for i = n - count + 1, n do t[i] = nil end
  table.setn(t, n - count)
  return unpack(removed, 1, count)
end

-- `string.byte(s [, i [, j]])`: each byte from i to j, negative positions from the end.
local nativeByte = string.byte
function string.byte(s, i, j)
  if j == nil then return nativeByte(s, i) end
  s = tostring(s)
  local len = string.len(s)
  i = tonumber(i) or 1
  j = tonumber(j) or i
  if i < 0 then i = len + i + 1 end
  if j < 0 then j = len + j + 1 end
  if i < 1 then i = 1 end
  if j > len then j = len end
  if i > j then return end
  if j - i >= 2000 then error("stack overflow (string slice too long)", 2) end
  local out = {}
  for k = i, j do out[k - i + 1] = nativeByte(s, k) end
  return unpack(out, 1, j - i + 1)
end
strbyte = string.byte

-- `string.gsub` with a table: the first capture, or the whole match, is the key; a nil or false
-- value keeps the match, another non-string, non-number value is an error.
local nativeGsub = string.gsub
function string.gsub(s, pattern, repl, n)
  if type(repl) ~= "table" then return nativeGsub(s, pattern, repl, n) end
  local anchored = string.sub(pattern, 1, 1) == "^"
  local wrapped = anchored and ("^(" .. string.sub(pattern, 2) .. ")") or ("(" .. pattern .. ")")
  return nativeGsub(s, wrapped, function(whole, first)
    local value = repl[first or whole]
    if value == nil or value == false then return whole end
    local t = type(value)
    if t ~= "string" and t ~= "number" then error("invalid replacement value (a " .. t .. ")", 3) end
    return value
  end, n)
end
gsub = string.gsub
