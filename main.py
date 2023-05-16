from fastapi import FastAPI
from codes import laCode, reNFACode, ndaDFACode, leftRCode, leftFCode, firstFollowCode

app = FastAPI()


@app.get("/")
async def root():
    return {
        "Index": {
            "LEXICAL ANALYZER": "LA",
            "RE TO NFA": "RENFA",
            "NFA to DFA": "NFATODFA",
            "Left Recursion": "LEFTR",
            "Left Factoring": "LEFTF",
            "First and Follow": "FANDF",
            "Postfix and Prefix": "PostPre",
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
