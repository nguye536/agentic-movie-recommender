"""Worker: runs all 20 prompts against a single folder's llm.py.

Usage: python run_one.py <folder_path>
Outputs JSON lines (one per prompt) to stdout.
"""
import sys, os, json, time, importlib.util, traceback

TEST_PROMPTS = [
    {"preferences": "something good for tonight",                                        "history": [], "history_ids": []},
    {"preferences": "I don't know, just surprise me",                                    "history": [], "history_ids": []},
    {"preferences": "I want to cry in a good way",                                       "history": [], "history_ids": []},
    {"preferences": "something cozy and comforting for a rainy Sunday",                  "history": [], "history_ids": []},
    {"preferences": "I need adrenaline - edge-of-seat stuff",                            "history": [], "history_ids": []},
    {"preferences": "something funny and lighthearted, had a rough week",                "history": [], "history_ids": []},
    {"preferences": "a slow-burn psychological thriller",                                "history": [], "history_ids": []},
    {"preferences": "a raunchy comedy like Superbad or The Hangover",                    "history": [], "history_ids": []},
    {"preferences": "an epic fantasy adventure with great world-building",               "history": [], "history_ids": []},
    {"preferences": "a gritty crime drama, something like The Wire",                     "history": [], "history_ids": []},
    {"preferences": "a foreign language film with stunning cinematography",              "history": [], "history_ids": []},
    {"preferences": "an indie character study, quiet and nuanced",                       "history": [], "history_ids": []},
    {"preferences": "an animated film that adults will love too",                        "history": [], "history_ids": []},
    {"preferences": "something by Christopher Nolan",                                    "history": [], "history_ids": []},
    {"preferences": "a film with Margot Robbie",                                         "history": [], "history_ids": []},
    {"preferences": "I love Marvel, want more superhero action",
     "history": ["The Avengers", "Avengers: Infinity War"], "history_ids": [24428, 299536]},
    {"preferences": "dark comedy with a twist ending",
     "history": ["Parasite", "Joker"],                      "history_ids": [496243, 475557]},
    {"preferences": "something romantic but not cheesy",                                 "history": [], "history_ids": []},
    {"preferences": "action but also thoughtful and quiet",                              "history": [], "history_ids": []},
    {"preferences": "scary but not too scary, no gore",                                  "history": [], "history_ids": []},
]

def main():
    folder = sys.argv[1]
    llm_path = os.path.join(folder, "llm.py")
    sys.path.insert(0, folder)
    os.chdir(folder)

    try:
        spec = importlib.util.spec_from_file_location("llm", llm_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        get_recommendation = mod.get_recommendation
    except Exception as e:
        print(json.dumps({"fatal": f"import failed: {e}", "traceback": traceback.format_exc()}), flush=True)
        return

    for i, p in enumerate(TEST_PROMPTS):
        t0 = time.perf_counter()
        try:
            result = get_recommendation(p["preferences"], p["history"], p["history_ids"])
            elapsed = time.perf_counter() - t0
            out = {
                "idx": i,
                "preferences": p["preferences"],
                "history_ids": p["history_ids"],
                "elapsed": round(elapsed, 2),
                "tmdb_id": result.get("tmdb_id") if isinstance(result, dict) else None,
                "description": result.get("description") if isinstance(result, dict) else None,
                "extra_keys": sorted(set(result.keys()) - {"tmdb_id", "description"}) if isinstance(result, dict) else [],
                "error": None,
            }
        except Exception as e:
            elapsed = time.perf_counter() - t0
            out = {
                "idx": i,
                "preferences": p["preferences"],
                "history_ids": p["history_ids"],
                "elapsed": round(elapsed, 2),
                "tmdb_id": None,
                "description": None,
                "extra_keys": [],
                "error": f"{type(e).__name__}: {e}",
            }
        print(json.dumps(out), flush=True)

if __name__ == "__main__":
    main()
