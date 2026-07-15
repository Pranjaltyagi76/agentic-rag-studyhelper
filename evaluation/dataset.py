"""Curated RAG evaluation set (Phase 8.5).

Every chunk is LABELLED `relevant: True/False` against its question, which lets us
compute retrieval precision/recall/F1 from **ground truth** rather than an LLM's
opinion. Cases are deliberately adversarial:

  - `distractor`    : obviously off-topic chunks that must be dropped
  - `near_miss`     : chunks on a *closely related but wrong* topic (the hard case —
                      embeddings rank these highly, so only grading rejects them)
  - `multi_hop`     : the answer needs TWO chunks combined
  - `unanswerable`  : the notes genuinely do not contain the answer; the system should
                      abstain rather than invent (hallucination probe)

`answerable=False` cases have no relevant chunks by construction.
"""

EVAL_SET = [
    # ---------------------------------------------------------------- simple + distractor
    {
        "id": "photosynthesis-distractor",
        "type": "distractor",
        "file_name": "bio.pdf",
        "answerable": True,
        "question": "Explain photosynthesis from my notes",
        "reference": (
            "Photosynthesis turns sunlight, water and CO2 into glucose and oxygen using "
            "chlorophyll; light reactions in the thylakoids make ATP/NADPH and the Calvin "
            "cycle fixes carbon into sugars."
        ),
        "chunks": [
            {"text": "Photosynthesis converts sunlight, water, and carbon dioxide into glucose and oxygen using chlorophyll in the chloroplasts.", "relevant": True},
            {"text": "The light-dependent reactions occur in the thylakoid membranes and produce ATP and NADPH. The Calvin cycle then fixes carbon dioxide into glucose.", "relevant": True},
            {"text": "The Eiffel Tower is a wrought-iron lattice tower in Paris, completed in 1889 and standing 330 metres tall.", "relevant": False},
            {"text": "Binary search halves a sorted interval repeatedly and runs in O(log n) time.", "relevant": False},
        ],
    },
    # ---------------------------------------------------------------- near-miss (hard)
    {
        "id": "mitosis-vs-meiosis-near-miss",
        "type": "near_miss",
        "file_name": "bio.pdf",
        "answerable": True,
        "question": "According to my notes, how many daughter cells does mitosis produce and are they identical?",
        "reference": "Mitosis produces two genetically identical diploid daughter cells.",
        "chunks": [
            {"text": "Mitosis produces two genetically identical diploid daughter cells from a single parent cell.", "relevant": True},
            # Near-miss: same domain, similar vocabulary, WRONG process.
            {"text": "Meiosis produces four genetically distinct haploid gametes and involves crossing over during prophase I.", "relevant": False},
            {"text": "Binary fission is how prokaryotes such as bacteria divide asexually.", "relevant": False},
        ],
    },
    # ---------------------------------------------------------------- multi-hop
    {
        "id": "newton-multi-hop",
        "type": "multi_hop",
        "file_name": "phys.pdf",
        "answerable": True,
        "question": "From my notes, state the second law and give the SI unit of force with its base units.",
        "reference": (
            "Newton's second law: net force = mass x acceleration (F = m*a). The SI unit of "
            "force is the newton (N), equal to 1 kg*m/s^2."
        ),
        "chunks": [
            # Needs BOTH of these combined to answer fully.
            {"text": "Newton's second law states that the net force on an object equals its mass times its acceleration: F = m * a.", "relevant": True},
            {"text": "The SI unit of force is the newton (N), defined as one kilogram metre per second squared (kg*m/s^2).", "relevant": True},
            {"text": "Newton's first law states that an object remains at rest or in uniform motion unless acted on by a net force.", "relevant": False},
            {"text": "The Calvin cycle fixes carbon dioxide into glucose inside chloroplasts.", "relevant": False},
        ],
    },
    # ---------------------------------------------------------------- distractor-heavy
    {
        "id": "binary-search-heavy-distractors",
        "type": "distractor",
        "file_name": "algo.pdf",
        "answerable": True,
        "question": "What is the time complexity of binary search according to my notes, and what does it require of the input?",
        "reference": "Binary search runs in O(log n) time and requires the input array to be sorted.",
        "chunks": [
            {"text": "Binary search finds a target in a sorted array by repeatedly halving the search interval, giving O(log n) time complexity. It requires the input to be sorted.", "relevant": True},
            {"text": "Bubble sort repeatedly swaps adjacent out-of-order elements and runs in O(n^2) time.", "relevant": False},
            {"text": "A hash table offers O(1) average-case lookup but does not maintain ordering.", "relevant": False},
            {"text": "Depth-first search explores as far as possible along each branch before backtracking.", "relevant": False},
            {"text": "Photosynthesis occurs in the chloroplasts of green plants.", "relevant": False},
        ],
    },
    # ---------------------------------------------------------------- unanswerable (hallucination probe)
    {
        "id": "unanswerable-enzyme",
        "type": "unanswerable",
        "file_name": "bio.pdf",
        "answerable": False,
        "question": "According to my notes, what is the optimal pH for the enzyme amylase?",
        "reference": "The notes do not contain this information; the system should say so rather than invent a value.",
        "chunks": [
            {"text": "Photosynthesis converts sunlight, water, and carbon dioxide into glucose and oxygen using chlorophyll.", "relevant": False},
            {"text": "Mitosis produces two genetically identical diploid daughter cells.", "relevant": False},
        ],
    },
    {
        "id": "unanswerable-complexity",
        "type": "unanswerable",
        "file_name": "algo.pdf",
        "answerable": False,
        "question": "According to my notes, what is the worst-case time complexity of quicksort?",
        "reference": "The notes do not mention quicksort; the system should say so rather than invent a complexity.",
        "chunks": [
            {"text": "Binary search finds a target in a sorted array by repeatedly halving the search interval, giving O(log n) time complexity.", "relevant": False},
            {"text": "Depth-first search explores as far as possible along each branch before backtracking.", "relevant": False},
        ],
    },
]


def relevant_texts(example: dict) -> set[str]:
    return {c["text"] for c in example["chunks"] if c["relevant"]}


def all_texts(example: dict) -> list[str]:
    return [c["text"] for c in example["chunks"]]
