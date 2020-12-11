from typing import List

def postprocess(postprocessing_options: List[str], **kwargs):
    if 'none' in postprocessing_options:
        return
        
    if 'cluster' in postprocessing_options:
        from pyscreener.postprocessing.cluster import cluster
        d_smi_score_clusters = cluster(**kwargs)
    else:
        d_smi_score_clusters = None

    if 'visualize' in postprocessing_options:
        from pyscreener.postprocessing.visualization import visualize
        visualize(d_smi_score_clusters=d_smi_score_clusters, **kwargs)