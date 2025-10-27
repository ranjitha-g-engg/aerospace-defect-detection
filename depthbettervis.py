# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 13:38:52 2025

@author: mjaugustin.P01
"""

import numpy as np
import open3d as o3d
import sklearn
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import time
import matplotlib.pyplot as plt

class AircraftDentDetector:
    def __init__(self):
        # Initialize parameters for dent detection
        self.min_points = 100  # Minimum points to consider a dent
        self.eps = 0.02       # DBSCAN epsilon parameter
        self.min_samples = 5  # DBSCAN minimum samples parameter
        
        # Load or create the model
        try:
            self.model = tf.keras.models.load_model('dent_model.h5')
        except:
            print("Creating new model...")
            self.model = self._create_model()

    def _create_model(self):
        """Create a simple neural network model for dent classification"""
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(3,)),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def load_point_cloud(self, file_path):
        """Load PLY file and return point cloud object"""
        try:
            print(f"Loading point cloud from {file_path}")
            pcd = o3d.io.read_point_cloud(file_path)
            if len(pcd.points) == 0:
                raise ValueError("Empty point cloud loaded")
            return pcd
        except Exception as e:
            print(f"Error loading point cloud: {str(e)}")
            return None

    def preprocess_point_cloud(self, pcd):
        """Clean and normalize point cloud data"""
        try:
            # Remove statistical outliers
            print("Removing outliers...")
            pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
            
            # Convert to numpy array and normalize
            points = np.asarray(pcd.points)
            scaler = StandardScaler()
            points_normalized = scaler.fit_transform(points)
            
            return points_normalized, scaler
        except Exception as e:
            print(f"Error in preprocessing: {str(e)}")
            return None, None

    def detect_dents(self, points_normalized):
        """Detect dents using clustering"""
        try:
            # Apply DBSCAN clustering
            clustering = DBSCAN(eps=self.eps, min_samples=self.min_samples).fit(points_normalized)
            labels = clustering.labels_

            # Process clusters to find dents
            unique_labels = set(labels)
            dents = []
            
            for label in unique_labels:
                if label != -1:  # Ignore noise points
                    cluster_points = points_normalized[labels == label]
                    if len(cluster_points) >= self.min_points:
                        dent_info = self._analyze_dent(cluster_points)
                        dents.append(dent_info)

            return dents
        except Exception as e:
            print(f"Error in dent detection: {str(e)}")
            return []

    def _analyze_dent(self, cluster_points):
        """Analyze characteristics of a single dent"""
        # Calculate center point
        center = np.mean(cluster_points, axis=0)
        
        # Find dent depth using principal component analysis
        covariance_matrix = np.cov(cluster_points.T)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
        normal_vector = eigenvectors[:, np.argmin(eigenvalues)]
        
        # Calculate maximum deviation from the fitted plane
        depths = np.abs(np.dot(cluster_points - center, normal_vector))
        max_depth = np.max(depths)
        
        return {
            'center': center,
            'depth': max_depth,
            'num_points': len(cluster_points)
        }

    def process_realtime(self, file_path):
        """Process point cloud data with timing"""
        start_time = time.time()
        
        # Load and process data
        pcd = self.load_point_cloud(file_path)
        if pcd is None:
            return []
            
        points_normalized, scaler = self.preprocess_point_cloud(pcd)
        if points_normalized is None:
            return []
            
        # Detect and analyze dents
        dents = self.detect_dents(points_normalized)
        
        # Transform coordinates back to original space
        for dent in dents:
            dent['center'] = scaler.inverse_transform([dent['center']])[0]
            dent['depth'] = dent['depth'] * np.sqrt(scaler.var_[2])
        
        processing_time = time.time() - start_time
        print(f"Processing completed in {processing_time:.2f} seconds")
        
        return dents

    def visualize_results(self, pcd, dents):
        """Create 3D visualization of results"""
        try:
            points = np.asarray(pcd.points)
            
            # Create 3D plot
            fig = plt.figure(figsize=(12, 10))
            ax = fig.add_subplot(111, projection='3d')
            
            # Plot original surface points
            ax.scatter(points[:, 0], points[:, 1], points[:, 2], 
                       c='blue', s=1, alpha=0.1, label='Surface')
            
            # Plot detected dents with enhanced visibility
            for i, dent in enumerate(dents):
                center = dent['center']
                depth = dent['depth']
                
                # Use a color gradient based on depth
                color = plt.cm.viridis(depth / max(d['depth'] for d in dents))  # Normalize depth for color mapping
                
                # Place the star marker above the dent
                ax.scatter(center[0], center[1], center[2] + 0.1,  # Adjust Z-coordinate
                           c=color, s=200, marker='*', label=f'Dent {i+1}')
                
                # Place the text above the dent
                ax.text(center[0], center[1], center[2] + 0.2,  # Adjust Z-coordinate for text
                        f'Dent {i+1}: Depth: {depth:.2f} mm', 
                        color='black', fontsize=12, ha='center', fontweight='bold')
            
            # Set plot title and labels
            ax.set_title('Aircraft Surface Dent Detection', fontsize=16)
            ax.set_xlabel('X-axis', fontsize=12)
            ax.set_ylabel('Y-axis', fontsize=12)
            ax.set_zlabel('Z-axis', fontsize=12)
            
            # Add a grid for better visualization
            ax.grid(True)
            
            plt.legend()
            plt.show()
        except Exception as e:
            print(f"Error in visualization: {str(e)}")

def main():
    # Initialize detector
    detector = AircraftDentDetector()
    
    # Replace with your PLY file path
    file_path = "your_file_path"
    
    try:
        # Process the point cloud
        pcd = detector.load_point_cloud(file_path)
        if pcd is not None:
            dents = detector.process_realtime(file_path)
            
            # Print results
            print(f"\nDetected {len(dents)} dents:")
            for i, dent in enumerate(dents, 1):
                print(f"\nDent {i}:")
                print(f"Location: X={dent['center'][0]:.2f}, "
                      f"Y={dent['center'][1]:.2f}, "
                      f"Z={dent['center'][2]:.2f}")
                print(f"Depth: {dent['depth']:.2f}mm")
                print(f"Number of points: {dent['num_points']}")
            
            # Show visualization
            detector.visualize_results(pcd, dents)
            
    except Exception as e:
        print(f"Error in main execution: {str(e)}")

if __name__ == "__main__":
    main()