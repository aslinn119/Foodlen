"""Implementation tests only; synthetic losses are not dataset experiment results."""
import json
import time
from pathlib import Path
import torch
from torch import nn
from model_v2 import CaptionModelV2


def main():
    torch.manual_seed(42)
    torch.set_num_threads(4)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    start = time.time()
    vocab = {'<PAD>': 0, '<BOS>': 1, '<EOS>': 2, '<UNK>': 3,
             'a': 4, 'plate': 5, 'of': 6, 'rice': 7}
    # Keep production defaults and spatial resolution; no pretrained download.
    model = CaptionModelV2(len(vocab), pretrained=False).to(device)
    images = torch.randn(2, 3, 224, 224, device=device)
    captions = torch.tensor([[1, 4, 5, 6, 7, 2], [1, 4, 7, 2, 0, 0]], device=device)
    lengths = torch.tensor([6, 4])
    model.train()
    logits, alpha = model(images, captions, lengths, return_attention=True)
    assert logits.shape == (2, 5, 8)
    assert alpha.shape == (2, 5, 49)
    assert torch.allclose(alpha.sum(-1), torch.ones(2, 5, device=device), atol=1e-5)
    assert torch.isfinite(logits).all() and (alpha >= 0).all()
    loss = nn.functional.cross_entropy(logits.reshape(-1, 8), captions[:, 1:].reshape(-1), ignore_index=0)
    loss.backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    assert model.decoder.attention.score.weight.grad.abs().sum() > 0
    # Single synthetic batch, cached image features: decoder regression check.
    model.zero_grad(set_to_none=True)
    model.eval()
    with torch.no_grad():
        regions, global_embedding = model.encoder(images)
    model.decoder.train()
    optimizer = torch.optim.Adam(model.decoder.parameters(), lr=.003)
    losses = []
    for _ in range(20):
        scores = model.decoder(regions, global_embedding, captions, lengths)
        value = nn.functional.cross_entropy(scores.reshape(-1, 8), captions[:, 1:].reshape(-1), ignore_index=0)
        optimizer.zero_grad(set_to_none=True)
        value.backward()
        nn.utils.clip_grad_norm_(model.decoder.parameters(), 5.)
        optimizer.step()
        losses.append(value.item())
    assert sum(losses[-5:])/5 < sum(losses[:5])/5
    model.eval()
    words, maps = model.greedy_decode(images, vocab, max_len=8, return_attention=True)
    assert len(words) == len(maps) == 2
    for tokens, weights in zip(words, maps):
        assert weights.shape == (len(tokens), 49)
    single = model.greedy_decode(images[:1], vocab, max_len=8)[0]
    inverse = {index: word for word, index in vocab.items()}
    beam = [inverse[i] for i in model.beam_decode(images[0], vocab, beam_size=1, max_len=8)]
    assert single == beam
    with torch.no_grad():
        regions, global_embedding = model.encoder(images[:1])
        cached_beam = model.beam_decode_features(regions, global_embedding, vocab, beam_size=1, max_len=8)
    assert model.beam_decode(images[0], vocab, beam_size=1, max_len=8) == cached_beam
    with torch.no_grad():
        model.decoder.fc.weight.zero_()
        model.decoder.fc.bias.fill_(-100)
        model.decoder.fc.bias[2] = 100
    words, maps = model.greedy_decode(images, vocab, return_attention=True)
    assert words == [[], []] and all(a.shape == (0, 49) for a in maps)
    assert model.beam_decode(images[0], vocab) == []
    result = dict(status='PASS',device=str(device),torch=torch.__version__,
                  model_logits_shape=list(logits.shape),attention_shape=list(alpha.shape),
                  synthetic_decoder_losses_20_steps=losses,
                  checks=['224px forward','attention sums to one','finite gradients',
                          'attention receives gradients','synthetic decoder optimization',
                          'word heatmap alignment','beam1 equals greedy',
                          'cached-feature beam matches standalone beam','EOS termination'],
                  seconds=time.time()-start,
                  limitation='Synthetic implementation checks only; not Food500Cap training or BLEU results.')
    target = Path(__file__).resolve().parents[1]/'checks'
    target.mkdir(exist_ok=True)
    (target/'model_tests.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2),flush=True)


if __name__ == '__main__':
    main()
