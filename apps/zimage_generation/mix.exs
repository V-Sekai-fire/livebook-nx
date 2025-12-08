defmodule ZImageGeneration.MixProject do
  use Mix.Project

  def project do
    [
      app: :zimage_generation,
      version: "0.1.0",
      elixir: "~> 1.17",
      start_permanent: Mix.env() == :prod,
      deps: deps(),
      releases: [
        zimage_generation: [
          steps: [:assemble, :tar],
          include_executables_for: [:unix, :windows]
        ]
      ]
    ]
  end

  def application do
    [
      extra_applications: [:logger]
    ]
  end

  defp deps do
    [
      # No deps needed - script uses Mix.install/1 at runtime
    ]
  end
end


