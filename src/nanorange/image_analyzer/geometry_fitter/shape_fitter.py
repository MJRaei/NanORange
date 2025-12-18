"""
Shape fitting orchestrator for analyzing colorized microscopy images.

Coordinates contour finding, circle/ellipse fitting, and result merging
to extract geometric features from segmented images.
"""

import csv
import os
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import numpy as np
from skimage import morphology, measure

from nanorange.image_analyzer.geometry_fitter.contour_finder import (
    ContourFinder, 
    ColorSegmentationResult
)
from nanorange.image_analyzer.geometry_fitter.fit_shapes.fit_circle import (
    CircleFitter,
    CircleParams
)
from nanorange.image_analyzer.geometry_fitter.fit_shapes.fit_ellipse import (
    EllipseFitter,
    EllipseParams
)
from nanorange.schemas.schemas import ShapeFittingConfig


@dataclass
class ShapeFittingResult:
    """Result container for shape fitting analysis."""
    circles: List[CircleParams] = field(default_factory=list)
    ellipses: List[EllipseParams] = field(default_factory=list)
    
    @property
    def total_shapes(self) -> int:
        """Total number of detected shapes."""
        return len(self.circles) + len(self.ellipses)
    
    def to_dict_list(self) -> List[dict]:
        """Convert results to list of dictionaries for CSV export."""
        rows = []
        
        for circle in self.circles:
            rows.append({
                "Type": "Circle",
                "X": circle.xc,
                "Y": circle.yc,
                "R_or_A": circle.r,
                "B": "",
                "Theta": ""
            })
        
        for ellipse in self.ellipses:
            rows.append({
                "Type": "Ellipse",
                "X": ellipse.xc,
                "Y": ellipse.yc,
                "R_or_A": ellipse.a,
                "B": ellipse.b,
                "Theta": ellipse.theta
            })
        
        return rows


class ShapeFitter:
    """
    Orchestrates shape fitting on colorized microscopy images.
    
    Performs color-based segmentation, contour extraction, and
    fits circles/ellipses to the detected boundaries.
    """
    
    def __init__(self, config: Optional[ShapeFittingConfig] = None):
        """
        Initialize the shape fitter.
        
        Args:
            config: Configuration parameters (uses defaults if None)
        """
        self.config = config or ShapeFittingConfig()
        
        self.contour_finder = ContourFinder(
            min_object_size=self.config.min_object_size
        )
        
        self.circle_fitter = CircleFitter(
            min_radius=self.config.min_radius,
            max_radius=self.config.max_radius,
            max_rel_error=self.config.circle_rel_error_thresh,
            min_arc_fraction=self.config.min_arc_fraction,
            min_contour_points=self.config.min_contour_points
        )
        
        self.ellipse_fitter = EllipseFitter(
            min_radius=self.config.min_radius,
            max_radius=self.config.max_radius,
            max_rel_error=self.config.ellipse_error_thresh,
            min_arc_fraction=self.config.min_arc_fraction,
            min_contour_points=self.config.min_contour_points
        )
    
    def fit_shapes(self, image_path: str) -> ShapeFittingResult:
        """
        Fit shapes to a colorized image.
        
        Args:
            image_path: Path to the colorized image
            
        Returns:
            ShapeFittingResult containing detected circles and ellipses
        """
        seg_result = self.contour_finder.color_segment(
            image_path, 
            n_colors=self.config.n_colors
        )
        
        all_circles: List[CircleParams] = []
        all_ellipses: List[EllipseParams] = []
        
        for cluster_id in range(self.config.n_colors):
            circles, ellipses = self._process_cluster(
                seg_result.cluster_map, 
                cluster_id
            )
            all_circles.extend(circles)
            all_ellipses.extend(ellipses)
        
        merged_circles = self.circle_fitter.merge(
            all_circles,
            dist_thresh=self.config.merge_dist_thresh,
            size_thresh=self.config.merge_size_thresh
        )
        
        merged_ellipses = self.ellipse_fitter.merge(
            all_ellipses,
            dist_thresh=self.config.merge_dist_thresh,
            size_thresh=self.config.merge_size_thresh
        )
        
        return ShapeFittingResult(
            circles=merged_circles,
            ellipses=merged_ellipses
        )
    
    def _process_cluster(
        self, 
        cluster_map: np.ndarray, 
        cluster_id: int
    ) -> Tuple[List[CircleParams], List[EllipseParams]]:
        """
        Process a single color cluster to find shapes.
        
        Args:
            cluster_map: Cluster assignment map from segmentation
            cluster_id: ID of the cluster to process
            
        Returns:
            Tuple of (circles, ellipses) found in this cluster
        """
        circles: List[CircleParams] = []
        ellipses: List[EllipseParams] = []
        
        mask = (cluster_map == cluster_id)
        mask = morphology.remove_small_objects(
            mask, 
            min_size=self.config.min_object_size
        )
        
        if not mask.any():
            return circles, ellipses
        
        contours = measure.find_contours(mask.astype(float), 0.5)
        
        for contour in contours:
            if len(contour) < self.config.min_contour_points:
                continue
            
            y, x = contour[:, 0], contour[:, 1]
            
            circle_result = self._try_fit_circle(x, y)
            if circle_result is not None:
                circles.append(circle_result)
                continue
            
            ellipse_result = self._try_fit_ellipse(x, y)
            if ellipse_result is not None:
                ellipses.append(ellipse_result)
        
        return circles, ellipses
    
    def _try_fit_circle(
        self, 
        x: np.ndarray, 
        y: np.ndarray
    ) -> Optional[CircleParams]:
        """
        Try to fit a circle to the contour points.
        
        Args:
            x: X coordinates
            y: Y coordinates
            
        Returns:
            CircleParams if valid fit, None otherwise
        """
        fit_result = self.circle_fitter.fit_and_validate(x, y)
        return fit_result.params if fit_result else None
    
    def _try_fit_ellipse(
        self, 
        x: np.ndarray, 
        y: np.ndarray
    ) -> Optional[EllipseParams]:
        """
        Try to fit an ellipse to the contour points.
        
        Args:
            x: X coordinates
            y: Y coordinates
            
        Returns:
            EllipseParams if valid fit, None otherwise
        """
        fit_result = self.ellipse_fitter.fit_and_validate(x, y)
        return fit_result.params if fit_result else None
    
    @staticmethod
    def save_results_csv(
        result: ShapeFittingResult, 
        output_path: str
    ) -> bool:
        """
        Save shape fitting results to a CSV file.
        
        Args:
            result: ShapeFittingResult to save
            output_path: Path for the output CSV file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            with open(output_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["Type", "X", "Y", "R_or_A", "B", "Theta"])
                
                for circle in result.circles:
                    writer.writerow([
                        "Circle", 
                        circle.xc, 
                        circle.yc, 
                        circle.r, 
                        "", 
                        ""
                    ])
                
                for ellipse in result.ellipses:
                    writer.writerow([
                        "Ellipse",
                        ellipse.xc,
                        ellipse.yc,
                        ellipse.a,
                        ellipse.b,
                        ellipse.theta
                    ])
            
            return True
        except Exception as e:
            print(f"   ❌ Error saving CSV: {str(e)}")
            return False
