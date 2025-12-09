#!/usr/bin/env elixir

# Test all generation scripts with minimal parameters
# Uses test fixtures from thirdparty directory

Mix.install([
  {:jason, "~> 1.4.4"}
])

IO.puts("""
=== Testing All Generation Scripts ===
Running each script with minimal parameters to verify they work correctly.
""")

# Test fixtures from thirdparty
test_fixtures = %{
  image: "thirdparty/9zs80jkckhrma0ctz4as2vw900.jpeg",
  video: "thirdparty/rhy08tw6k9rma0ctz7m9y0xmgr.mp4",
  usdc: "thirdparty/monkey.usdc",
  glb: "thirdparty/UniRig/examples/giraffe.glb",
  audio: "thirdparty/kvoicewalk/example/target.wav"
}

# Verify fixtures exist
IO.puts("\n=== Verifying Test Fixtures ===")
fixtures_ok = Enum.all?(test_fixtures, fn {name, path} ->
  exists = File.exists?(path)
  status = if exists, do: "✓", else: "✗"
  IO.puts("  #{status} #{name}: #{path}")
  exists
end)

if !fixtures_ok do
  IO.puts("\n⚠ Some test fixtures are missing. Tests will be skipped.")
  System.halt(1)
end

# Test configurations - minimal parameters for each script
tests = [
  %{
    name: "qwen3vl_inference.exs",
    command: ~s(elixir qwen3vl_inference.exs "#{test_fixtures.image}" "what is this" --max-tokens 128),
    description: "Qwen3-VL vision-language inference (minimal tokens)"
  },
  %{
    name: "partcrafter_generation.exs",
    command: ~s(elixir partcrafter_generation.exs "#{test_fixtures.image}" --num-steps 5 --num-parts 2),
    description: "PartCrafter 3D mesh generation (5 steps, 2 parts)"
  },
  %{
    name: "zimage_generation.exs",
    command: ~s(elixir zimage_generation.exs "a simple test image" --steps 2 --width 256 --height 256),
    description: "ZImage generation (2 steps, 256x256)"
  },
  %{
    name: "sam3_video_segmentation.exs",
    command: ~s(elixir sam3_video_segmentation.exs "#{test_fixtures.video}"),
    description: "SAM3 video segmentation"
  },
  %{
    name: "tris_to_quads_converter.exs",
    command: ~s(elixir tris_to_quads_converter.exs "#{test_fixtures.usdc}"),
    description: "Triangles to quads converter"
  },
  %{
    name: "unirig_generation.exs",
    command: ~s(elixir unirig_generation.exs "#{test_fixtures.glb}"),
    description: "UniRig skeleton generation"
  },
  %{
    name: "kvoicewalk_generation.exs",
    command: ~s(elixir kvoicewalk_generation.exs --target-audio "#{test_fixtures.audio}" --target-text "Hello world" --step-limit 5 --population-limit 2),
    description: "KVoiceWalk voice cloning (5 steps, 2 population)"
  },
  %{
    name: "kokoro_tts_generation.exs",
    command: ~s(elixir kokoro_tts_generation.exs "Hello, this is a test."),
    description: "Kokoro TTS generation"
  },
  %{
    name: "corrective_smooth_baker.exs",
    command: ~s(elixir corrective_smooth_baker.exs "#{test_fixtures.usdc}"),
    description: "Corrective smooth baker"
  }
]

IO.puts("\n=== Running Tests ===\n")

results = Enum.map(tests, fn test ->
  IO.puts("Testing: #{test.name}")
  IO.puts("  Description: #{test.description}")
  IO.puts("  Command: #{test.command}")
  IO.puts("  Running...")

  start_time = System.monotonic_time(:second)

  result = case System.cmd("powershell", ["-Command", test.command],
    stderr_to_stdout: true
  ) do
    {output, exit_code} ->
      duration = System.monotonic_time(:second) - start_time
      status = if exit_code == 0, do: "✓ PASS", else: "✗ FAIL"

      IO.puts("  #{status} (exit code: #{exit_code}, duration: #{duration}s)")

      # Show last few lines of output if failed
      if exit_code != 0 do
        lines = String.split(output, "\n") |> Enum.filter(&(&1 != ""))
        last_lines = Enum.take(lines, -5)
        if last_lines != [] do
          IO.puts("  Last output lines:")
          Enum.each(last_lines, fn line -> IO.puts("    #{String.slice(line, 0, 100)}") end)
        end
      end

      %{test: test.name, status: if(exit_code == 0, do: :pass, else: :fail), exit_code: exit_code, duration: duration}
  end

  IO.puts("")
  result
end)

IO.puts("\n=== Test Summary ===")
passed = Enum.count(results, fn r -> r.status == :pass end)
total = length(results)
failed = total - passed

IO.puts("Total: #{total}")
IO.puts("Passed: #{passed}")
IO.puts("Failed: #{failed}")

if failed > 0 do
  IO.puts("\nFailed tests:")
  Enum.each(results, fn r ->
    if r.status == :fail do
      IO.puts("  ✗ #{r.test} (exit code: #{r.exit_code})")
    end
  end)
end

IO.puts("\n=== Detailed Results ===")
Enum.each(results, fn r ->
  status_icon = if r.status == :pass, do: "✓", else: "✗"
  IO.puts("#{status_icon} #{r.test}: #{r.status} (#{r.duration}s)")
end)

if failed == 0 do
  IO.puts("\n✓ All tests passed!")
  System.halt(0)
else
  IO.puts("\n✗ Some tests failed")
  System.halt(1)
end
