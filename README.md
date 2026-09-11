# GNN-Based BERT for Understanding Context from Music

This project implements a hybrid BERT + Graph Neural Network (GNN) system to understand music by combining text (metadata/tags) and audio structure (chroma/mel-spectrograms).

## Setup
It is recommended to run the initial preprocessing locally and the training on Google Colab if your PC is not powerful.

1. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Workflow
1. **Download Data**: Place the FMA-small dataset in `data/raw/`.
2. **Preprocess Audio**: Run `src/audio_features.py` to extract features.
3. **Build Graphs**: Run `src/graph_builder.py` to construct segment graphs.
4. **Train**: Run `src/train.py` locally or upload the `gnn-bert-music-context` folder to Google Colab and run it there.

## Project Structure
- `config.yaml`: All hyperparameters and file paths.
- `src/`: Core implementation files for Task 1 (BERT), Task 2 (GNN), Task 3 (Fusion), and Task 4 (Contrastive).
- `notebooks/`: For EDA and demoing the end-to-end model.
