"""Turn the exported chat zip into a readable AI-log.md (just your messages and Claude's replies)."""
import glob, json, os, re, sys, zipfile

downloads = os.path.expanduser("~/Downloads")
zips = sorted(glob.glob(os.path.join(downloads, "session-export-*.zip")), key=os.path.getmtime)
if not zips:
    sys.exit("No session-export-*.zip found in Downloads")
zip_path = zips[-1]

with zipfile.ZipFile(zip_path) as z:
    # main transcript = biggest .jsonl that isn't a subagent transcript
    names = [n for n in z.namelist() if n.endswith(".jsonl") and "subagent" not in n.lower()]
    main = max(names, key=lambda n: z.getinfo(n).file_size)
    lines = z.read(main).decode("utf-8").splitlines()

hidden = re.compile(r"<system-reminder>.*?</system-reminder>|<pasted_content[^>]*>|</pasted_content[^>]*>", re.S)

def text_of(content):
    if isinstance(content, str):
        return content
    parts = []
    for block in content or []:
        if isinstance(block, dict) and block.get("type") == "text":
            parts.append(block.get("text", ""))
        elif isinstance(block, dict) and block.get("type") == "image":
            parts.append("[screenshot]")
    return "\n".join(parts)

out = ["# AI log: Catalyst AI/ML Week 1, Part 1\n"]
for line in lines:
    try:
        entry = json.loads(line)
    except json.JSONDecodeError:
        continue
    if entry.get("type") not in ("user", "assistant") or entry.get("isMeta"):
        continue
    msg = entry.get("message", {})
    text = hidden.sub("", text_of(msg.get("content"))).strip()
    if not text or text == "[screenshot]" and entry["type"] == "assistant":
        continue
    who = "Akari" if entry["type"] == "user" else "Claude"
    out.append(f"---\n\n**{who}:**\n\n{text}\n")

dest = os.path.join(os.path.dirname(os.path.abspath(__file__)), "AI-log.md")
with open(dest, "w", encoding="utf-8") as f:
    f.write("\n".join(out))
print("Wrote", dest)
