import open_clip
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
clip_model_name = "ViT-L-14"
clip_model,_,clip_preprocess = open_clip.create_model_and_transforms(clip_model_name
clip_model_name,pretrained = "openai",precision='fp16' if device == 'cuda' else 'fp32',device=device,
)

tokenize = open_clip.get_tokenizer(clip_model_name)


def image_to_features(image: Image.Image) -> torch.Tensor:
    images = clip_preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad(), torch.cuda.amp.autocast():
        image_features = clip_model.encode_image(images)
    return image_features


img = cv.imread("/path/to/example.png")
img = Image.fromarray(img)

image_feature = image_to_features(img)
print(image_feature.shape)

prompt = "a photo of a cat"
text_tokens = tokenize([prompt]).to(device)
text_features = clip_model.encode_text(text_tokens)
print(text_features.shape)