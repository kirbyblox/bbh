## Bridge Bid Hallucinator
bbh-amber.vercel.app

Bridge bot that hallucinates bids based on past bids.

Model is a simple character level RNN trained on ~ 200 bidding sequences that runs client-side.


Currently WIP
To do:
- More sophisticated sampling
- Clean up UI
- Incorporate more bridge rules and bids
- Make website mobile friendly
- Figure out how to minify js without breaking model
- Think of a good way of using hand data


## To Run Locally:

```bash
npm run dev
```