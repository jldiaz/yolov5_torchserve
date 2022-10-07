import io
import os
import sys
import time
import asyncio
import httpx
import concurrent

from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse, RedirectResponse

ALLOWED_MODELS = { 
    's': "yolov5s", 
#   "x": "yolov5x"
}
ALLOWED_TYPES = {'jpg', 'jpeg', 'png', 'webp'}
TORCHSERVE_URL=os.environ.get("TOCHSERVE_URL", "http://localhost:8080/predictions")


# This could be an external file
FORM_TEMPLATE = '''
<!doctype html>
<head>
<title>FastYolo5 - Image upload</title>
</head>
<body>
<h1>FastYolo5 - Image upload</h1>
<form method=post enctype=multipart/form-data target=_blank>
    <input type="file" name="file" id="file">
    <input type=submit value=Upload>

 <p/>
  <label for="model">Choose Yolo model:</label>
  <select name="model" id="model">
    <option value="x">Yolov5x</option>
    <option value="s">Yolov5s</option>
  </select>

 <p/>
  <label for="threads">OpenMP Num Threads:</label>
  <input type="number" id=threads name=threads min='0'>
  
    <input type=checkbox id=augment name=augment value=true>
    <label for=augment>Augment</label>

    <p>Result type</p>
    <input type=radio id=imagen name=tipo value=imagen checked>
    <label for=imagen>Labelled image</label>
    <input type=radio id=json name=tipo value=json> 
    <label for=json>JSON</label>

</form>
</body>
'''


# Create app
app = FastAPI()
print(f"Using {TORCHSERVE_URL} as inference server.", file=sys.stderr)

# Utility functions
def allowed_file(mime):
    """Checks if the extension of the file sent by the client is valid"""
    type, subtype = mime.split("/")
    if type!="image":
        return False
    if subtype not in ALLOWED_TYPES:
        return False
    return True


# async def run_in_bg(pool, f):
#     """Function to asynchronously run another function in a different thread"""
#     loop = asyncio.get_event_loop()
#     result = await loop.run_in_executor(pool, f)
#     return result

# async def detect_and_label_image(file_contents, m, augment):
#     """Calls Yolo detector on the input image and produces a labelled image"""
#     _in = await run_in_bg(preprocessing_pool, lambda: cv2.imdecode(np.asarray(bytearray(file_contents), dtype=np.uint8),1))
#     _out = None
#     r = await run_in_bg(inference_pool, lambda: detect(_in, model[m], augment))
#     img = await run_in_bg(preprocessing_pool, lambda:  label_image(_in, r, _out))
#     return img

# async def detect_and_get_meta(file_contents, m, augment):
#     """Calls Yolo detector on the input image and returns a Python dictionary with the results"""

#     _in = await run_in_bg(preprocessing_pool, lambda: cv2.imdecode(np.asarray(bytearray(file_contents), dtype=np.uint8),1))
#     r = await run_in_bg(inference_pool, lambda: detect(_in, model[m], augment))
#     return get_meta(r)

async def do_inference(model, img_file) -> dict:
    url = f"{TORCHSERVE_URL}/{ALLOWED_MODELS[model]}"
    t1 = time.perf_counter()
    http_client = httpx.AsyncClient()
    r = await http_client.post(url,
              files={'img1': img_file})
    t2 = time.perf_counter()
    result = r.json()
    # result["meta"] = {}
    # result["meta"]["time"] = t2-t1
    return result

# FASTAPI routes
@app.get("/")
async def index_redirect():
    return RedirectResponse("/detect")

@app.get('/detect')
async def get_detect_form():
    return HTMLResponse(content=FORM_TEMPLATE, status_code=200)

@app.post("/detect")
async def process_detect_form(file: UploadFile = File(), 
            model: str = Form("s"), threads: int = Form(0),
            augment: str = Form("no"), tipo: str = Form()):
    t1 = time.time()
    ok = allowed_file(file.content_type)
    if not ok:
        raise HTTPException(status_code=422, detail="File type not supported (only JPG or PNG).")
    # file_content = await file.read()
    if model not in ALLOWED_MODELS:
        msg = f"You cannot use model {model!r}, only {list(ALLOWED_MODELS)}"
        raise HTTPException(status_code=501, detail=msg)
    
    # augment not supported
    # if augment.lower() in ("false", "0", "no"):
    #    augment = False
    # augment = bool(augment)

    # Thread set not allowed
    # try:
    #     threads = int(threads)
    # except:
    #     threads = 0
    # if threads <= 0:
    #     autoset_threads()
    # else:
    #     torch.set_num_threads(threads)

    if tipo.lower() != "json":
        msg = "Image labelling not yet supported. Please select json output"
        raise HTTPException(status_code=501, detail=msg)
        # img = await detect_and_label_image(file_content, model, augment)
        # def send_file(img):
        #     yield from io.BytesIO(img)
        # return StreamingResponse(send_file(img), media_type="image/jpg")
    else:
        metadata = await do_inference(model, file.file)
        t2 = time.time()
        metadata["meta"]["totaltime"] = t2-t1
        metadata["meta"]["model"] = model
        return JSONResponse(metadata)


