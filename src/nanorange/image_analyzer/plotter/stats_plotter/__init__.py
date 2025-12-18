"""
Stats plotter module for visualizing statistical data from detected shapes.

Provides tools for creating histograms and distribution plots.
"""

from nanorange.image_analyzer.plotter.stats_plotter.size_distribution import (
    SizeDistributionPlotter,
    SizeDistributionConfig,
    plot_size_distribution
)

__all__ = [
    "SizeDistributionPlotter",
    "SizeDistributionConfig",
    "plot_size_distribution",
]
