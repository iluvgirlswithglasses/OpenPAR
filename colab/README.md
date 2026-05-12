
This machine is not capable of training these models. To train a model (like UniPAR) do the following:
1. Upload the code to Google Drive via `./upload.sh UniPAR`
2. Use Colab MCP to get the code, the dataset, and the necessary checkpoints (see `./UniPAR_setup.ipynb` for example)

**NEW**: The checkpoint `PETA.pth` which was mentioned in `MSP60K_Benchmark_Dataset/readme.md` can be copied into Google Colab runtime via:

```
!cp drive/MyDrive/yomikawa-reid/ckpt/LLM-PAR-PETA.pth
```

