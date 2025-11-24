from datetime import datetime
from wellness_log import WellnessEntry, append_entry, load_last_entry

if __name__ == "__main__":
    e = WellnessEntry(
        timestamp=datetime.now().isoformat(timespec="seconds"),
        mood="test-mood",
        energy="medium",
        stresses="just testing",
        goals=["finish setup"],
        self_care=["drink water"],
        summary="Test entry for the wellness log.",
    )

    print("Appending entry…")
    path = append_entry(e)
    print("Saved to:", path)

    print("Last entry:", load_last_entry())
