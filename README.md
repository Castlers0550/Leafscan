# 🌿 LeafScan AI



The app lets you upload an image and returns the top 3 most likely diagnoses with confidence scores. The model was trained on over 70,000 images covering 38 disease classes across 14 different plant types.

I built this project to learn more about computer vision, model deployment, and how machine learning models can be turned into tools that real people can use.

##  Stats

- 99.9% Validation Accuracy
- 70,000+ Training Images
- 38 Disease Classes
- 14 Plant Types


##  What I Used

- Python
- TensorFlow
- EfficientNetB0
- Flask
- HTML
- CSS
- JavaScript
- Google Colab
- Render
- Vercel

##  How It Works

1. Upload a photo of a plant leaf.
2. The image is resized and preprocessed.
3. The image is sent to a Flask API.
4. An EfficientNetB0 model analyzes the image.
5. The top 3 predictions are returned with confidence scores.

##  Example Output

```text
1. Apple - Cedar Apple Rust (99.7%)
2. Apple - Scab (0.2%)
3. Apple - Healthy (0.1%)
```

##  What I Learned

This project taught me a lot about:

- Training image classification models
- Preventing overfitting
- Deploying machine learning models
- Building REST APIs
- Connecting frontends and backends
- Working with large datasets

One challenge was making sure the preprocessing pipeline used during training matched the preprocessing used during inference. Even small differences could noticeably affect the model's predictions.

##  Future Improvements

- More plant species
- Disease treatment recommendations
- Mobile app support
- Real-time camera scanning
- Explainable AI visualizations

##  Author

Eben Siyabalapitiya
