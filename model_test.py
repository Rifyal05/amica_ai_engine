from llama_cpp import Llama
import time

print("Test Start...")
llm = Llama(
    model_path="./models/gemma-3-1b-it-q4_km.gguf",
    n_ctx=2048,
    n_threads=4, 
    verbose=False
)

start = time.time()
output = llm(
    "<start_of_turn>user\nBerikan tips parenting singkat untuk anak tantrum.<end_of_turn>\n<start_of_turn>model\n",
    max_tokens=100,
    stop=["<end_of_turn>"]
)
end = time.time()

print("\nHASIL TES")
print(output["choices"][0]["text"]) # type: ignore
print(f"\nWaktu total: {end - start:.2f} detik")