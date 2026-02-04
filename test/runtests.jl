using Test
using ESMFold

@testset "Alphabet" begin
    alphabet = Alphabet_from_architecture("ESM-1b")
    @test length(alphabet.all_toks) == 33
    @test alphabet.all_toks[1] == "<cls>"
    @test alphabet.all_toks[end] == "<mask>"
    @test alphabet.tok_to_idx["<mask>"] == alphabet.mask_idx
end

@testset "Sequence mapping" begin
    @test sequence_to_af2_indices("ACD") == [0, 4, 3]
end

@testset "Padding helper" begin
    seqs = [[0, 1, 2], [3]]
    aa, mask = ESMFold._pad_batch(seqs; pad_value = 0)
    @test size(aa) == (2, 3)
    @test size(mask) == (2, 3)
    @test aa[2, 2] == 0
    @test mask[2, 2] == 0
end

@testset "Embedding forward shapes" begin
    alphabet = Alphabet_from_architecture("ESM-1b")
    esm = ESM2(1, 16, 4; alphabet = alphabet, token_dropout = false)
    model = ESMFoldEmbed(esm; c_s = 8, c_z = 1, use_esm_attn_map = false)

    emb = model(["AC", "M"])
    @test size(emb) == (8, 2, 2)

    out = model(["AC"]; return_pair = true)
    @test out.pair === nothing
    @test size(out.sequence) == (8, 2, 1)
end
