# Robust Graph Recommendation via Noise-Aware Adversarial Perturbation [DASFAA '24]

1. Download datasets MovieLense 1M (ML-1M), Amazon Toys and Games (Amazon-Toys) and 2022-Yelp(Yelp), respectively. Put These into data/raw-ml-1M, data/raw-amazon_toys, data/pro-yelp, respectively.
   URL:
   1. ML-1M: [https://drive.google.com/file/d/1sqgFpwHNWNPaMlVFHbQQXIRaAN9i3KUJ/view?usp=drive_link]()
   2. Amazon-Toys: [https://drive.google.com/file/d/1De099aEeHZ-8rKcscElYnO1Mo3pCZ-oO/view?usp=drive_link]()
   3. Yelp: [https://drive.google.com/file/d/1o79AJLqrDGGixzfNayczunuoSNQ7hCSm/view?usp=drive_link]()
2. Use *.ipynb to preprocess the data: (1) xxx clean dataset: clean-xxx.ipynb (2) xxx noisy data: Noise-xxx.ipynb
3. ```
   python code/quick_start.py --dataset ml-1M
   ```
