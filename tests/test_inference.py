from sdg_clf import inference, utils, modelling
import os
os.chdir("..")

text = """
    Between 2014 and the onset of the pandemic, the number of people going hungry and suffering from food insecurity had been gradually rising. The COVID-19 crisis has pushed those rising rates even higher. The war in Ukraine is further disrupting global food supply chains and creating the biggest global food crisis since the Second World War. The COVID-19 crisis has also exacerbated all forms of malnutrition, particularly in children.

In 2020, between 720 and 811 million persons worldwide were suffering from hunger, as many as 161 million more than in 2019. Also in 2020, over 30 per cent – a staggering 2.4 billion people – were moderately or severely food-insecure, lacking regular access to adequate food. This represents an increase of almost 320 million people in the course of just one year.

Globally, 149.2 million children under five years of age, or 22.0 per cent, were suffering from stunting (low height for age) in 20202, the proportion having decreased from 24.4 per cent in 2015. These numbers may become higher, however, owing to continued constraints on accessing nutritious diets and essential nutrition services during the pandemic, with the full impact possibly taking years to manifest itself. To achieve the target of a 5 per cent reduction in the number of stunted children by 2025, the current rate of decline of 2.1 per c
    """
tokenizer = utils.get_tokenizer("roberta-large")
input_creator = inference.TextToModelInputs(tokenizer)
def test_prepare_input_ids():
    input_ids = input_creator.prepare_input_ids(text)
    assert len(input_ids) > 1


def test_prepare_attention_mask():
    input_ids = input_creator.prepare_input_ids(text)
    attention_mask = input_creator.prepare_attention_mask(input_ids)
    assert len(attention_mask) == len(input_ids)

def test_prepare_model_inputs():
    model_inputs = input_creator.prepare_model_inputs(text)
    input_ids = model_inputs["input_ids"]
    attention_mask = model_inputs["attention_mask"]
    assert len(input_ids) == len(attention_mask)
    assert input_ids[0].ndim == 2

def test_predict_on_sample():
    onnx_model = modelling.create_model_for_provider("sdg_clf/finetuned_models/roberta-large_1608124504.onnx")
    tokenizer = utils.get_tokenizer("roberta-large")
    sdgs = inference.predict_on_sample(text, onnx_model, tokenizer)