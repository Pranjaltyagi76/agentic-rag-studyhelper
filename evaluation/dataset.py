"""Curated RAG evaluation set (Phase 8.5).

Each example seeds a few note chunks, asks a question those notes should answer, and
gives a reference answer used as the gold standard by the judges. Small on purpose —
enough to detect regressions when a prompt or retrieval knob changes.
"""

EVAL_SET = [
    {
        "id": "photosynthesis",
        "file_name": "bio.pdf",
        "chunks": [
            "Photosynthesis converts sunlight, water, and carbon dioxide into glucose and "
            "oxygen using chlorophyll in the chloroplasts.",
            "The light-dependent reactions occur in the thylakoid membranes and produce ATP "
            "and NADPH. The Calvin cycle then fixes carbon dioxide into glucose.",
            "The Eiffel Tower is a wrought-iron tower in Paris completed in 1889.",  # distractor
        ],
        "question": "Explain photosynthesis from my notes",
        "reference": (
            "Photosynthesis is how plants turn sunlight, water, and CO2 into glucose and "
            "oxygen via chlorophyll. It has light-dependent reactions (in the thylakoids, "
            "making ATP/NADPH) and the Calvin cycle (fixing CO2 into glucose)."
        ),
    },
    {
        "id": "newtons-second-law",
        "file_name": "phys.pdf",
        "chunks": [
            "Newton's second law states that the net force on an object equals its mass times "
            "its acceleration: F = m * a.",
            "The SI unit of force is the newton (N), equal to one kg*m/s^2.",
            "Photosynthesis occurs in plants.",  # distractor
        ],
        "question": "State Newton's second law from my notes and its unit of force",
        "reference": (
            "Newton's second law: the net force equals mass times acceleration (F = m*a). "
            "The SI unit of force is the newton (N) = kg*m/s^2."
        ),
    },
    {
        "id": "binary-search",
        "file_name": "algo.pdf",
        "chunks": [
            "Binary search finds a target in a sorted array by repeatedly halving the search "
            "interval, giving O(log n) time complexity.",
            "It requires the input to be sorted; otherwise the result is undefined.",
        ],
        "question": "How does binary search work and what is its time complexity, from my notes?",
        "reference": (
            "Binary search works on a sorted array by repeatedly halving the search interval "
            "to locate a target, with O(log n) time complexity. It requires sorted input."
        ),
    },
]
