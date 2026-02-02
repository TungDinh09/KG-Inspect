import re

pattern = re.compile(r'(^|,)\s*[^,]*bot[^,]*\s*(,|$)', re.IGNORECASE)

tests = [
    None,
    "",
    "alice",
    "alice,bob",
    "project_3135_bot_d01eab66a4f70a82355365511d731ca5",
    "alice,project_3135_bot_d01eab66a4f70a82355365511d731ca5",
    "project_3135_bot_d01eab66a4f70a82355365511d731ca5,alice",
    "alice,deploy-bot,bob",
    "cicd-commit",
]

for value in tests:
    if value is None or value == "":
        print(f"{value!r:55} -> MATCH=False (no assignees, allow pipeline)")
        continue

    match = bool(pattern.search(value))
    print(f"{value!r:55} -> MATCH={match}  -> {'BLOCK pipeline' if match else 'ALLOW pipeline'}")
