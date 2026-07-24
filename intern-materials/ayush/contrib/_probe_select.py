from pathlib import Path
p = Path("/home/user/Desktop/Ayush PH test/venv/lib/python3.12/site-packages/lerobot/policies/smolvla/modeling_smolvla.py")
text = p.read_text()
for key in ["def select_action", "def predict_action_chunk", "def predict_action"]:
    i = text.find(key)
    print("====", key, "at", i)
    if i >= 0:
        print(text[i:i + 1200])
        print()
