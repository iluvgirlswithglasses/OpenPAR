import os

# Override any path by setting the corresponding env var.
# Defaults point to the standard Colab layout used by LLM-PAR_setup.ipynb.
_MODEL_ROOT = os.environ.get('LLMPAR_MODEL_ROOT', '/content/models')
_DATA_ROOT  = os.environ.get('LLMPAR_DATA_ROOT',  '/content')

google_bert_path  = os.environ.get('LLMPAR_BERT_PATH',     'bert-base-uncased')
minigpt4_path     = os.environ.get('LLMPAR_MINIGPT4_PATH',  os.path.join(_MODEL_ROOT, 'pretrained_minigpt4_7b.pth'))
vicuna_7b_path    = os.environ.get('LLMPAR_VICUNA_PATH',    os.path.join(_MODEL_ROOT, 'vicuna-7b-v1.5'))
blip2_path        = os.environ.get('LLMPAR_BLIP2_PATH',     os.path.join(_MODEL_ROOT, 'blip2_pretrained_flant5xxl.pth'))
eva_vit_g_path    = os.environ.get('LLMPAR_EVA_VIT_PATH',   os.path.join(_MODEL_ROOT, 'eva_vit_g.pth'))


def get_pkl_rootpath(dataset):
    if dataset == "RAPv1":
        root_path = os.path.join(_DATA_ROOT, 'RAP', 'RAP_dataset')
        pkl_path  = os.path.join(_DATA_ROOT, 'RAP', 'rap1_template.pkl')
    elif dataset == "RAPv2":
        root_path = os.path.join(_DATA_ROOT, 'RAPV2', 'RAP_dataset')
        pkl_path  = os.path.join(_DATA_ROOT, 'RAPV2', 'rap2_template.pkl')
    elif dataset == "PETA":
        root_path = os.path.join(_DATA_ROOT, 'PETA', 'images')
        pkl_path  = os.path.join(_DATA_ROOT, 'PETA', 'peta_template.pkl')
    elif dataset == "PA100k":
        root_path = os.path.join(_DATA_ROOT, 'PA100K', 'release_data', 'release_data')
        pkl_path  = os.path.join(_DATA_ROOT, 'PA100K', 'pa100k_template.pkl')
    elif dataset == "RAPzs":
        root_path = os.path.join(_DATA_ROOT, 'RAPV2', 'RAP_dataset')
        pkl_path  = os.path.join(_DATA_ROOT, 'RAPV2', 'rapzs_template.pkl')
    elif dataset == "PETAzs":
        root_path = os.path.join(_DATA_ROOT, 'PETA', 'images')
        pkl_path  = os.path.join(_DATA_ROOT, 'PETA', 'petazs_template.pkl')
    elif dataset == "MSP":
        root_path = os.path.join(_DATA_ROOT, 'MSP60k', 'SUBMIT', 'images')
        pkl_path  = os.path.join(_DATA_ROOT, 'MSP60k', 'SUBMIT', 'msp_random_template.pkl')
    elif dataset == "MSPCD":
        root_path = os.path.join(_DATA_ROOT, 'MSP60k', 'SUBMIT', 'images')
        pkl_path  = os.path.join(_DATA_ROOT, 'MSP60k', 'SUBMIT', 'msp_cd_template.pkl')
    return pkl_path, root_path
