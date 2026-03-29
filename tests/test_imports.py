# tests/test_imports.py
def test_imports():
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    
    # Проверяем, что все модули грузятся
    from src import features_main
    from src import episodes
    from src import augmentation
    # import visualize_data
    
    # Проверяем, что ключевые функции существуют
    assert callable(features_main.add_time_features)
    assert callable(episodes.make_episodes)
    assert callable(augmentation.jitter)
    # assert callable(visualize_data.plot_correlation_matrix)
