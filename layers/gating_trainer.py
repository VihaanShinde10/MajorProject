import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
import json
from pathlib import Path
from sklearn.model_selection import train_test_split
from .layer6_gating import GatingNetwork

class GatingDataset(Dataset):
    """Dataset for training gating network."""
    
    def __init__(self, features: np.ndarray, targets: np.ndarray):
        """
        Args:
            features: (N, 8) array of gating features
            targets: (N,) array of optimal alpha values
        """
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]

class GatingTrainer:
    """
    Trainer for the gating network.
    Learns optimal alpha (text vs behavior weight) from labeled data.
    """
    
    def __init__(self, model: GatingNetwork = None, device: str = 'cpu'):
        self.model = model if model else GatingNetwork()
        self.device = torch.device(device)
        self.model.to(self.device)
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001, weight_decay=1e-5)
        self.criterion = nn.MSELoss()
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        
        self.train_losses = []
        self.val_losses = []
    
    def prepare_training_data(self, 
                             transactions_df: pd.DataFrame,
                             results_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare training data from transaction history.
        
        Args:
            transactions_df: DataFrame with transaction data
            results_df: DataFrame with classification results including:
                       - text_confidence, behavior_confidence
                       - correct_category (ground truth)
                       - predicted_category
                       
        Returns:
            features: (N, 8) array of gating features
            targets: (N,) array of optimal alpha values
        """
        features_list = []
        targets_list = []
        
        for idx, row in results_df.iterrows():
            # Extract features
            text_conf = row.get('text_confidence', 0.5)
            behavior_conf = row.get('behavior_confidence', 0.5)
            token_count = row.get('token_count', 3)
            is_generic = row.get('is_generic_text', False)
            recurrence_conf = row.get('recurrence_confidence', 0.0)
            cluster_density = row.get('cluster_density', 0.0)
            user_txn_count = row.get('user_txn_count', 0)
            semantic_consensus = row.get('semantic_consensus', 0.0)
            
            features = np.array([
                text_conf,
                behavior_conf,
                token_count / 10.0,  # Normalize
                1.0 if is_generic else 0.0,
                recurrence_conf,
                cluster_density,
                min(user_txn_count / 100.0, 1.0),  # Normalize
                semantic_consensus
            ])
            
            # Compute optimal alpha (ground truth)
            correct_category = row.get('correct_category', row.get('true_category'))
            text_prediction = row.get('text_prediction')
            behavior_prediction = row.get('behavior_prediction')
            
            if correct_category:
                # Optimal alpha: favor the method that predicted correctly
                if text_prediction == correct_category and behavior_prediction != correct_category:
                    optimal_alpha = 0.85  # Trust text
                elif behavior_prediction == correct_category and text_prediction != correct_category:
                    optimal_alpha = 0.15  # Trust behavior
                elif text_prediction == correct_category and behavior_prediction == correct_category:
                    # Both correct - weight by confidence
                    optimal_alpha = text_conf / (text_conf + behavior_conf + 1e-6)
                    optimal_alpha = np.clip(optimal_alpha, 0.15, 0.85)
                else:
                    # Both wrong - use confidence-based weighting
                    optimal_alpha = 0.5  # Neutral
                
                features_list.append(features)
                targets_list.append(optimal_alpha)
        
        return np.array(features_list), np.array(targets_list)
    
    def train_epoch(self, dataloader: DataLoader) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        
        for features, targets in dataloader:
            features = features.to(self.device)
            targets = targets.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            predictions = self.model(features).squeeze()
            
            # Compute loss
            loss = self.criterion(predictions, targets)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(dataloader)
    
    def validate(self, dataloader: DataLoader) -> float:
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for features, targets in dataloader:
                features = features.to(self.device)
                targets = targets.to(self.device)
                
                predictions = self.model(features).squeeze()
                loss = self.criterion(predictions, targets)
                
                total_loss += loss.item()
        
        return total_loss / len(dataloader)
    
    def train(self, 
              features: np.ndarray, 
              targets: np.ndarray,
              epochs: int = 100,
              batch_size: int = 32,
              val_split: float = 0.2,
              early_stopping_patience: int = 15) -> Dict:
        """
        Train the gating network.
        
        Args:
            features: (N, 8) array of gating features
            targets: (N,) array of optimal alpha values
            epochs: Number of training epochs
            batch_size: Batch size
            val_split: Validation split ratio
            early_stopping_patience: Early stopping patience
            
        Returns:
            Training history
        """
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            features, targets, test_size=val_split, random_state=42
        )
        
        # Create datasets
        train_dataset = GatingDataset(X_train, y_train)
        val_dataset = GatingDataset(X_val, y_val)
        
        # Create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        
        print(f"Training gating network on {len(X_train)} samples...")
        print(f"Validation set: {len(X_val)} samples")
        
        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)
            
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            
            # Learning rate scheduling
            self.scheduler.step(val_loss)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                self.save_checkpoint('models/gating_best.pt')
            else:
                patience_counter += 1
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} - "
                      f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            if patience_counter >= early_stopping_patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        # Load best model
        self.load_checkpoint('models/gating_best.pt')
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': best_val_loss,
            'epochs_trained': epoch + 1
        }
    
    def save_checkpoint(self, path: str):
        """Save model checkpoint."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }, path)
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])

def train_gating_from_history(history_csv: str, 
                              output_model_path: str = 'models/gating_trained.pt',
                              epochs: int = 100) -> Dict:
    """
    Train gating network from historical transaction data.
    
    Args:
        history_csv: Path to CSV with columns:
                    - text_confidence, behavior_confidence
                    - token_count, is_generic_text
                    - recurrence_confidence, cluster_density
                    - user_txn_count, semantic_consensus
                    - text_prediction, behavior_prediction
                    - correct_category (ground truth)
        output_model_path: Where to save trained model
        epochs: Number of training epochs
        
    Returns:
        Training history
    """
    # Load data
    df = pd.read_csv(history_csv)
    
    # Initialize trainer
    trainer = GatingTrainer()
    
    # Prepare data
    features, targets = trainer.prepare_training_data(df, df)
    
    print(f"Prepared {len(features)} training samples")
    print(f"Target alpha range: [{targets.min():.2f}, {targets.max():.2f}]")
    print(f"Target alpha mean: {targets.mean():.2f}")
    
    # Train
    history = trainer.train(features, targets, epochs=epochs)
    
    # Save final model
    trainer.save_checkpoint(output_model_path)
    print(f"Trained model saved to {output_model_path}")
    
    return history

if __name__ == '__main__':
    # Example usage
    import sys
    
    if len(sys.argv) > 1:
        history_csv = sys.argv[1]
        history = train_gating_from_history(history_csv)
        print(f"Training complete. Best validation loss: {history['best_val_loss']:.4f}")
    else:
        print("Usage: python gating_trainer.py <history_csv>")
        print("CSV should contain transaction features and ground truth labels")

