from fastapi import FastAPI
from codes import (
    laCode,
    reNFACode,
    ndaDFACode,
    leftRCode,
    leftFCode,
    firstFollowCode,
    postpCODE,
    dagCode,
    leadTrailCode,
)

app = FastAPI()


@app.get("/")
async def root():
    return {
        "Index": {
            "LEXICAL ANALYZER": "la",
            "RE TO NFA": "renfa",
            "NFA to DFA": "nfadfa",
            "Left Recursion": "leftr",
            "Left Factoring": "leftf",
            "First and Follow": "firstf",
            "Postfix and Prefix": "postp",
            "Construction of Dag": "dag",
            "lead and trail": "leadt",
        }
    }


@app.get("/la")
async def la():
    return {"code": laCode}


@app.get("/renfa")
async def renfa():
    return {"code": reNFACode}


@app.get("/nfadfa")
async def nfaDfa():
    return {"code": ndaDFACode}


@app.get("/leftr")
async def leftr():
    return {"code": leftRCode}


@app.get("/leftf")
async def leftf():
    return {"code": leftFCode}


@app.get("/firstf")
async def firstFollow():
    return {"code": firstFollowCode}


@app.get("/postp")
async def postp():
    return {"code": postpCODE}


@app.get("/dag")
async def dag():
    return {"code": dagCode}


@app.get("/leadt")
async def leadtrail():
    return {"code": leadTrailCode}


import uvicorn

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
