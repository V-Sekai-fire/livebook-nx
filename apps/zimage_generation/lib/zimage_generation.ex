defmodule ZImageGeneration do
  @moduledoc """
  Entry point that executes the original script.
  The script stays as a .exs file and uses Mix.install/1 at runtime.
  """

  def main(args) do
    # Get the script path from priv directory
    script_path = Path.join([:code.priv_dir(:zimage_generation), "zimage_generation.exs"])

    # Execute the script with provided args
    # This preserves the exact behavior of running: elixir zimage_generation.exs <args>
    Code.eval_file(script_path, args)
  end
end


