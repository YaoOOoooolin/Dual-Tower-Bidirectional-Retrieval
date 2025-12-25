# Dual-Tower-Bidirectional-Retrieval
SaProt-650M (protein tower) + ChemBERTa (molecule tower) + projection alignment + FAISS bidirectional retrieval + cold-target split + test Recall@K

# Task 1 Report: Dual-Tower Bidirectional Retrieval (Protein ↔ Molecule) in Colab

## 1. Objective
The goal of Task 1 is to build a **Dual-Tower retrieval model** that supports **two-way retrieval**:

1) **Protein → Molecule**: given a protein target, retrieve/recommend relevant small molecules.  
2) **Molecule → Protein**: given a molecule (SMILES), retrieve potential protein targets.

The assignment requires the protein tower to use **SaProt (650M or 1.3B)**, while the molecular tower can be selected from the literature (e.g., GNN, Transformer, ChemBERTa). The implementation should be runnable in Colab (ColabService-style).

---

## 2. Dataset and Preprocessing

### 2.1 Dataset
I used the HuggingFace dataset **`eve-bio/drug-target-activity`** (loaded via streaming). Each record includes protein-compound assay information. For retrieval supervision, I extracted:

- `target__uniprot_id` (protein ID)
- `compound__smiles` (molecule)
- `outcome_is_active` (activity label)
- `outcome_potency_pxc50` (potency value; logged but not directly used as the training label)

### 2.2 Positive Pair Construction
To create positive pairs for contrastive learning, I treated records with `outcome_is_active == True` as **positive protein–molecule pairs**. From the streamed subset, I obtained:

- **Total unique positive pairs**: 4915  
- **Unique proteins**: 171  
- **Unique molecules (SMILES)**: 558  

### 2.3 Protein Sequence Retrieval
SaProt requires **protein sequences**. The dataset provides UniProt IDs but not always sequences, so I fetched sequences from UniProt FASTA endpoints using `requests`, parsed FASTA, and merged them back into the pair table. Pairs without valid sequences were filtered out.

---

## 3. Method

### 3.1 Dual-Tower Architecture
The system consists of two independent encoders (“towers”) producing dense embeddings.

#### Protein Tower (Mandatory: SaProt 650M)
- Model: `westlake-repl/SaProt_650M_AF2`
- Input: amino-acid sequence
- Tokenization: AA-only SaProt format (each amino acid followed by `#`)
- Representation: mean pooling over the last hidden states
- Output dimension: **1280**

#### Molecular Tower (Chosen: ChemBERTa)
- Model: `DeepChem/ChemBERTa-77M-MLM`
- Input: SMILES string
- Representation: mean pooling over the last hidden states
- Output dimension: **384**

### 3.2 Projection to a Shared Embedding Space
Because the two towers output different dimensions (1280 vs 384), I added lightweight trainable **projection heads**:

- `proj_p: R^1280 → R^256` (protein projection)
- `proj_m: R^384  → R^256` (molecule projection)

After projection, embeddings are **L2-normalized**, so inner product corresponds to cosine similarity. The shared retrieval dimension is **d = 256**.

### 3.3 Training Objective (Symmetric InfoNCE with In-Batch Negatives)
I trained only the projection heads (the large encoders remain frozen as feature extractors), using a symmetric contrastive loss:

- For a batch of matched pairs `(p_i, m_i)`, compute  
  `logits = (p @ m^T) / temperature`
- Use cross-entropy so that each `p_i` matches `m_i` among in-batch negatives, and vice-versa:
  `loss = 0.5 * (CE(logits, labels) + CE(logits^T, labels))`

Temperature was set to **0.07**.

---

## 4. Retrieval System Implementation

### 4.1 Embedding Precomputation and Caching
To make the notebook runnable in Colab (even on CPU), I precomputed and cached base embeddings:

- Protein base embeddings: `prot_base` with shape `(171, 1280)`
- Molecule base embeddings: `mol_base` with shape `(558, 384)`

These were saved as `.npy` files so repeated runs do not re-encode all sequences/SMILES.

### 4.2 FAISS Indexing
After projection and normalization, I built two FAISS indexes using **inner-product** similarity:

- Molecule index: `mol_index = faiss.IndexFlatIP(256)`  
- Protein index: `prot_index = faiss.IndexFlatIP(256)`

### 4.3 Bidirectional Search APIs
I implemented two query functions:

- `search_mols_by_protein(uniprot_id, topk)` → returns Top-K SMILES for a protein
- `search_prots_by_smiles(smiles, topk)` → returns Top-K UniProt IDs for a molecule

Both use FAISS retrieval in the shared embedding space.

---

## 5. Experimental Setup

### 5.1 Cold-Target Train/Test Split
To evaluate generalization to **unseen protein targets**, I used a **protein-level split**:

- Train proteins: 136  
- Test proteins: 35  
- Train pairs: 3686  
- Test pairs: 909  

This “cold-target” setup prevents the model from seeing test proteins during training.

### 5.2 Training Hyperparameters (Colab CPU)
Because the run was done in a constrained environment, I used a simple and stable configuration:

- Optimizer: AdamW  
- Learning rate: 1e-3, weight decay: 1e-4  
- Training steps: 800  
- Batch size: 128  
- Temperature: 0.07  

Only the projection heads were trained.

---

## 6. Evaluation and Results

### 6.1 Metric: Recall@K (Protein → Molecule)
For each test protein, I retrieved Top-K molecules. If **any** ground-truth positive molecule for that protein appears in Top-K, it is counted as a hit:

`Recall@K = (#hit proteins) / (#test proteins)`

### 6.2 Cold-Target Test Results (Protein → Molecule)
On the cold-target test proteins (35 unseen proteins), I obtained:

- **Test P→M Recall@1  = 0.3429**
- **Test P→M Recall@5  = 0.5714**
- **Test P→M Recall@10 = 0.7429**

These results show that even with frozen encoders and only lightweight projection training, the model retrieves correct molecules at useful rates in a realistic unseen-target setting (higher K gives higher recall as expected).

### 6.3 Bidirectional Functionality (Molecule → Protein)
The Molecule → Protein retrieval path was implemented symmetrically using the protein FAISS index and was validated by returning Top-K targets for a given SMILES (e.g., the correct protein target appears among top results in the demo queries). The same Recall@K evaluation can be applied to M→P using the test pairs.

---

## 7. Conclusion
I successfully implemented Task 1’s required **Dual-Tower bidirectional retrieval system** in Colab:

1) Used **SaProt-650M** as the mandatory protein encoder and **ChemBERTa** as the molecular encoder.  
2) Added trainable **projection heads** to align protein and molecule embeddings into a shared 256-D space.  
3) Built **FAISS** indexes for efficient retrieval and implemented both **Protein→Molecule** and **Molecule→Protein** search.  
4) Evaluated under a **cold-target split** and achieved:
   - Recall@1 = 0.3429  
   - Recall@5 = 0.5714  
   - Recall@10 = 0.7429  

---

## 8. Online Demo (Interactive)
To demonstrate the system in an interactive “online demo” format, I implemented a lightweight web UI (ColabService-style) using **Gradio**. The demo provides two interactive tabs:

- **Protein → Molecule**: input/select a UniProt ID and retrieve Top-K SMILES.
- **Molecule → Protein**: input/select a SMILES and retrieve Top-K UniProt IDs.

Because this demo is launched from Colab during runtime (with a shareable web link), it supports real-time interaction while the code is running.  
A screen recording of the interaction workflow is provided as **`demo.mp4`**, showing example queries and retrieval results for both directions.
