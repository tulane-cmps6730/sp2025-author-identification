# Authorship Verification: Comparing Modern Models


Ever wondered if two different documents were written by the same person? From forensic linguistics trying to identify anonymous authors to plagiarism detection, figuring out who wrote what is a fascinating challenge!


Our project investigates and compares different generations of AI models to tackle the authorship verification task: determining if different text samples originate from the same author based purely on writing style. We aim to provide a clear comparison between modern Transformer-based approaches and older methods.


**The Goals We Had:**


* Investigate and compare the effectiveness of modern Transformer models (like BERT) against older recurrent architectures (RNNs, LSTMs) and traditional methods for authorship verification.
* Evaluate the performance of these models using standard NLP metrics like F1-score, precision, recall, and accuracy.
* Demonstrate the capabilities and nuances of recent deep learning advancements in capturing unique authorial styles.

**The Methods We Used:**
* Utilized powerful pre-trained Transformer models (specifically BERT variants) accessed via the Hugging Face `transformers` library.
* Implemented a Siamese-like architecture where known and unknown texts are encoded using the same BERT model.
* Derived text embeddings from the [CLS] token's output and averaged embeddings when multiple texts were provided per sample.
* Created a comparison feature vector by concatenating the known embedding, unknown embedding, and their absolute difference.
* Built the model, data handling (Datasets, DataLoaders), and training loop using PyTorch.
* Used datasets from authorship competitions (PAN 2013, PAN 2015) for training and evaluation.


**Conclusion**
Our project explored the challenging task of authorship verification by comparing various model architectures. By leveraging the power of pre-trained Transformers like BERT and comparing them against other methods, we gained insights into how different techniques capture the subtle signals of writing style. While challenges like data preparation and hyperparameter tuning exist, modern deep learning models show significant improvement for this task.
For more details, check out the full project report (`report/Report.pdf`).
For Screenshots of our demo, check out the screenshots in the report
