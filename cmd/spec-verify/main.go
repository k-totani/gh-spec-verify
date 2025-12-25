package main

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"strings"

	"github.com/k-totani/spec-verify/internal/config"
	"github.com/k-totani/spec-verify/internal/verifier"
)

const version = "0.1.0"

func main() {
	if len(os.Args) < 2 {
		printUsage()
		os.Exit(0)
	}

	command := os.Args[1]

	switch command {
	case "init":
		runInit()
	case "check", "verify":
		runCheck(os.Args[2:])
	case "version", "-v", "--version":
		fmt.Printf("spec-verify version %s\n", version)
	case "help", "-h", "--help":
		printUsage()
	default:
		// コマンドなしで直接タイプ指定の場合
		runCheck(os.Args[1:])
	}
}

func printUsage() {
	fmt.Println(`spec-verify - SPEC駆動開発のための検証ツール

Usage:
  spec-verify <command> [options]

Commands:
  init          設定ファイルを初期化
  check [type]  SPECとコードの一致度を検証
                type: ui, api, または省略で全て
  version       バージョンを表示
  help          このヘルプを表示

Options:
  --format json    JSON形式で出力（CI向け）
  --threshold N    合格ラインを指定（デフォルト: 50）
  --config FILE    設定ファイルを指定

Environment Variables:
  ANTHROPIC_API_KEY    Claude APIキー
  OPENAI_API_KEY       OpenAI APIキー
  GOOGLE_API_KEY       Gemini APIキー
  SPEC_VERIFY_API_KEY  汎用APIキー

Examples:
  spec-verify init
  spec-verify check
  spec-verify check ui
  spec-verify check --format json
  spec-verify check api --threshold 70`)
}

func runInit() {
	configFile := config.FindConfigFile()

	if _, err := os.Stat(configFile); err == nil {
		fmt.Printf("設定ファイル %s は既に存在します。上書きしますか？ [y/N] ", configFile)
		var answer string
		fmt.Scanln(&answer)
		if strings.ToLower(answer) != "y" {
			fmt.Println("キャンセルしました。")
			return
		}
	}

	cfg := config.DefaultConfig()
	if err := cfg.Save(configFile); err != nil {
		fmt.Printf("エラー: 設定ファイルの作成に失敗しました: %v\n", err)
		os.Exit(1)
	}

	fmt.Printf("✅ 設定ファイル %s を作成しました。\n", configFile)
	fmt.Println("\n次のステップ:")
	fmt.Println("1. 設定ファイルを編集してプロジェクトに合わせてください")
	fmt.Println("2. ANTHROPIC_API_KEY 環境変数を設定してください")
	fmt.Println("3. specs/ ディレクトリにSPECファイルを配置してください")
	fmt.Println("4. spec-verify check を実行してください")
}

func runCheck(args []string) {
	var specType string
	var jsonOutput bool
	var threshold int
	var configFile string

	// 引数をパース
	for i := 0; i < len(args); i++ {
		arg := args[i]
		switch {
		case arg == "--format" && i+1 < len(args):
			if args[i+1] == "json" {
				jsonOutput = true
			}
			i++
		case arg == "--threshold" && i+1 < len(args):
			fmt.Sscanf(args[i+1], "%d", &threshold)
			i++
		case arg == "--config" && i+1 < len(args):
			configFile = args[i+1]
			i++
		case !strings.HasPrefix(arg, "-"):
			specType = arg
		}
	}

	// 設定を読み込む
	if configFile == "" {
		configFile = config.FindConfigFile()
	}

	cfg, err := config.Load(configFile)
	if err != nil {
		fmt.Printf("エラー: 設定ファイルの読み込みに失敗しました: %v\n", err)
		os.Exit(1)
	}

	// オプションをオーバーライド
	if threshold > 0 {
		cfg.Options.PassThreshold = threshold
	}

	// APIキーの確認
	if cfg.AIAPIKey == "" {
		fmt.Println("エラー: APIキーが設定されていません。")
		fmt.Println("ANTHROPIC_API_KEY 環境変数を設定するか、設定ファイルに api_key を追加してください。")
		os.Exit(1)
	}

	// Verifierを作成
	v, err := verifier.New(cfg)
	if err != nil {
		fmt.Printf("エラー: Verifierの作成に失敗しました: %v\n", err)
		os.Exit(1)
	}

	// 検証を実行
	ctx := context.Background()

	if !jsonOutput {
		fmt.Println("\n🔍 SPEC検証を開始します...\n")
		fmt.Println(strings.Repeat("━", 50))
	}

	summary, err := v.VerifyAll(ctx, specType)
	if err != nil {
		fmt.Printf("エラー: 検証に失敗しました: %v\n", err)
		os.Exit(1)
	}

	if jsonOutput {
		outputJSON(summary)
	} else {
		outputConsole(summary)
	}

	// 終了コード
	if !summary.IsPassing(cfg.Options.PassThreshold) {
		os.Exit(1)
	}
}

func outputJSON(summary *verifier.Summary) {
	data, _ := json.MarshalIndent(summary, "", "  ")
	fmt.Println(string(data))
}

func outputConsole(summary *verifier.Summary) {
	for _, result := range summary.Results {
		fmt.Printf("\n📄 %s\n", result.SpecFile)
		fmt.Printf("   タイトル: %s\n", result.Title)
		if result.RoutePath != "" {
			fmt.Printf("   パス: %s\n", result.RoutePath)
		}
		fmt.Printf("   関連コード: %dファイル\n", len(result.CodeFiles))

		if result.Error != nil {
			fmt.Printf("   ❌ エラー: %v\n", result.Error)
			continue
		}

		if result.Verification == nil {
			fmt.Println("   ⚠️  検証結果がありません")
			continue
		}

		emoji := getMatchEmoji(result.Verification.MatchPercentage)
		fmt.Printf("   %s 一致度: %d%%\n", emoji, result.Verification.MatchPercentage)

		if len(result.Verification.MatchedItems) > 0 {
			fmt.Println("   ✓ 一致:")
			for i, item := range result.Verification.MatchedItems {
				if i >= 3 {
					fmt.Printf("     ... 他%d件\n", len(result.Verification.MatchedItems)-3)
					break
				}
				fmt.Printf("     - %s\n", item)
			}
		}

		if len(result.Verification.UnmatchedItems) > 0 {
			fmt.Println("   ✗ 不一致:")
			for i, item := range result.Verification.UnmatchedItems {
				if i >= 3 {
					fmt.Printf("     ... 他%d件\n", len(result.Verification.UnmatchedItems)-3)
					break
				}
				fmt.Printf("     - %s\n", item)
			}
		}
	}

	// サマリー
	fmt.Println("\n" + strings.Repeat("━", 50))
	fmt.Println("\n📊 サマリー\n")
	fmt.Printf("   総SPEC数: %d\n", summary.TotalSpecs)
	fmt.Printf("   平均一致度: %.1f%%\n", summary.AverageMatch)
	fmt.Printf("   高一致(≥80%%): %d件\n", summary.HighMatchCount)
	fmt.Printf("   低一致(<50%%): %d件\n", summary.LowMatchCount)

	// 詳細バー
	fmt.Println("\n   詳細:")
	for _, result := range summary.Results {
		percentage := 0
		if result.Verification != nil {
			percentage = result.Verification.MatchPercentage
		}
		bar := strings.Repeat("█", percentage/10) + strings.Repeat("░", 10-percentage/10)
		fmt.Printf("   %s %3d%% %s\n", bar, percentage, result.SpecFile)
	}

	fmt.Println()
}

func getMatchEmoji(percentage int) string {
	if percentage >= 80 {
		return "✅"
	} else if percentage >= 50 {
		return "⚠️"
	}
	return "❌"
}
