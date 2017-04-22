# Sentiment-Long-ShortTermMemoryNN

Architecture:
                    "Positive"
                        |
  Sigmoid   Sigmoid  Sigmoid
    |         |         |
   LSTM ->  LSTM  ->   LSTM
    |         |         |
Embedding Embedding Embedding
    |         |         |
   input     input     input
   
   - Used embeddings to prevent one-hot encoding of 1000s of classes
   - Validation Accuracy ~85% at 10 epochs
   - Test Accuracy ~83%
