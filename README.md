# LLM-Explorer
A repository to explore the latest developments in LLMs and run experiments!

1. `model_fetcher.py`
Fetches an model from the the HuggingFace library and runs an inference through it
2. `generate_synthetic_data.py`
Generates artifical data from the input content prompt provided using llamafiles version of TinyLlama-1.1Bmodel.
```
    Example of Input:
        name,region
        Rahul,India
        Mark,America
        Schneider,Germany
        OKenny,Ireland
        Mieke,Germany
```  
The output generates is 
```  
        1. Name: Rahul
        Region: India
        2. Name: Mark
        Region: America
        3. Name: Schneider
        Region: Germany
        4. Name: OKenny
        Region: Ireland
        5. Name: Mikee
        Region: Germany
        6. Name: Miek
        Region: Germany
        7. Name: Jake
        Region: United States
        8. Name: Sarah
        Region: United Kingdom
        9. Name: John
        Region: United Kingdom
        10. Name: Tom
        Region: United Kingdom
```  
