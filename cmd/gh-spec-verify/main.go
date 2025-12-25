package main

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"strings"

	"github.com/k-totani/gh-spec-verify/internal/ai"
	"github.com/k-totani/gh-spec-verify/internal/config"
	"github.com/k-totani/gh-spec-verify/internal/parser"
	"github.com/k-totani/gh-spec-verify/internal/verifier"
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
	case "endpoints":
		runEndpoints(os.Args[2:])
	case "coverage":
		runCoverage(os.Args[2:])
	case "version", "-v", "--version":
		fmt.Printf("gh-spec-verify version %s\n", version)
	case "help", "-h", "--help":
		printUsage()
	default:
		// コマンドなしで直接タイプ指定の場合
		runCheck(os.Args[1:])
	}
}

// commonOptions holds common command-line options for multiple commands
type commonOptions struct {
	jsonOutput bool
	configFile string
	// check-specific options
	threshold int
	failUnder int
	specType  string
}

// parseCommonOptions parses common options from arguments
func parseCommonOptions(args []string) commonOptions {
	var opts commonOptions

	for i := 0; i < len(args); i++ {
		arg := args[i]
		switch {
		case arg == "--format" && i+1 < len(args):
			if args[i+1] == "json" {
				opts.jsonOutput = true
			}
			i++
		case arg == "--config" && i+1 < len(args):
			opts.configFile = args[i+1]
			i++
		case arg == "--threshold" && i+1 < len(args):
			fmt.Sscanf(args[i+1], "%d", &opts.threshold)
			i++
		case arg == "--fail-under" && i+1 < len(args):
			fmt.Sscanf(args[i+1], "%d", &opts.failUnder)
			i++
		case !strings.HasPrefix(arg, "-"):
			// Non-flag argument (e.g., spec type for check command)
			if opts.specType == "" {
				opts.specType = arg
			}
		}
	}

	return opts
}

func printUsage() {
	fmt.Println(`gh-spec-verify - SPEC駆動開発のための検証ツール (GitHub CLI Extension)

Usage:
  gh spec-verify <command> [options]

Commands:
  init          設定ファイルを初期化
  check [type]  SPECとコードの一致度を検証
                type: ui, api, または省略で全て
  endpoints     APIエンドポイント一覧を表示
  coverage      APIカバレッジレポートを表示
  version       バージョンを表示
  help          このヘルプを表示

Options:
  --format json    JSON形式で出力（CI向け）
  --threshold N    合格ラインを指定（デフォルト: 50）
  --fail-under N   個別閾値を指定（N%未満のSPECがあれば失敗）
  --config FILE    設定ファイルを指定

Environment Variables:
  ANTHROPIC_API_KEY    Claude APIキー
  OPENAI_API_KEY       OpenAI APIキー
  GOOGLE_API_KEY       Gemini APIキー
  SPEC_VERIFY_API_KEY  汎用APIキー

Examples:
  gh spec-verify init
  gh spec-verify check
  gh spec-verify check ui
  gh spec-verify check --format json
  gh spec-verify check api --threshold 70
  gh spec-verify coverage
  gh spec-verify coverage --format json`)
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
	fmt.Println("4. gh spec-verify check を実行してください")
}

func runCheck(args []string) {
	// Parse all options including check-specific ones
	commonOpts := parseCommonOptions(args)

	// 設定を読み込む
	configFile := commonOpts.configFile
	if configFile == "" {
		configFile = config.FindConfigFile()
	}

	cfg, err := config.Load(configFile)
	if err != nil {
		fmt.Printf("エラー: 設定ファイルの読み込みに失敗しました: %v\n", err)
		os.Exit(1)
	}

	// オプションをオーバーライド
	if commonOpts.threshold > 0 {
		cfg.Options.PassThreshold = commonOpts.threshold
	}
	if commonOpts.failUnder > 0 {
		cfg.Options.FailUnder = commonOpts.failUnder
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

	if !commonOpts.jsonOutput {
		fmt.Println("\n🔍 SPEC検証を開始します...\n")
		fmt.Println(strings.Repeat("━", 50))
	}

	summary, err := v.VerifyAll(ctx, commonOpts.specType)
	if err != nil {
		fmt.Printf("エラー: 検証に失敗しました: %v\n", err)
		os.Exit(1)
	}

	// 個別閾値チェック
	if cfg.Options.FailUnder > 0 {
		summary.FailUnder = cfg.Options.FailUnder
		summary.FailingSpecs = buildFailingSpecs(summary.Results, cfg.Options.FailUnder)
	}

	if commonOpts.jsonOutput {
		outputJSON(summary)
	} else {
		outputConsole(summary, cfg.Options.FailUnder)
	}

	// 終了コード
	failed := false
	if !summary.IsPassing(cfg.Options.PassThreshold) {
		failed = true
	}
	if len(summary.FailingSpecs) > 0 {
		failed = true
	}
	if failed {
		os.Exit(1)
	}
}

// buildFailingSpecs は個別閾値を下回ったSPECを抽出する
func buildFailingSpecs(results []verifier.Result, failUnder int) []verifier.FailingSpec {
	var failing []verifier.FailingSpec
	for _, result := range results {
		// エラーがあるものは対象外（エラーは別で表示）
		if result.Error != nil {
			continue
		}
		if result.Verification != nil && result.Verification.MatchPercentage < failUnder {
			failing = append(failing, verifier.FailingSpec{
				SpecFile:        result.SpecFile,
				Title:           result.Title,
				MatchPercentage: result.Verification.MatchPercentage,
			})
		}
	}
	return failing
}

func outputJSON(summary *verifier.Summary) {
	data, _ := json.MarshalIndent(summary, "", "  ")
	fmt.Println(string(data))
}

func outputConsole(summary *verifier.Summary, failUnder int) {
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

		emoji := getStatusEmoji(float64(result.Verification.MatchPercentage))
		// 個別閾値未達の場合はマークを追加
		belowThreshold := ""
		if failUnder > 0 && result.Verification.MatchPercentage < failUnder {
			belowThreshold = fmt.Sprintf(" ← Below threshold (%d%%)", failUnder)
		}
		fmt.Printf("   %s 一致度: %d%%%s\n", emoji, result.Verification.MatchPercentage, belowThreshold)

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

	// 個別閾値未達の表示
	if len(summary.FailingSpecs) > 0 {
		fmt.Printf("\n❌ 個別閾値未達 (%d%% 未満): %d件\n", failUnder, len(summary.FailingSpecs))
		for _, spec := range summary.FailingSpecs {
			fmt.Printf("   - %s (%d%%) : %s\n", spec.SpecFile, spec.MatchPercentage, spec.Title)
		}
	}

	fmt.Println()
}

// getStatusEmoji returns an emoji based on the percentage threshold
func getStatusEmoji(percentage float64) string {
	if percentage >= 80 {
		return "✅"
	} else if percentage >= 50 {
		return "⚠️"
	}
	return "❌"
}

// loadConfigAndProvider loads config and creates AI provider from common options
// Returns config, provider, and bool indicating success (false means error was printed and os.Exit should be called)
func loadConfigAndProvider(opts commonOptions) (*config.Config, ai.Provider, bool) {
	configFile := opts.configFile
	if configFile == "" {
		configFile = config.FindConfigFile()
	}

	cfg, err := config.Load(configFile)
	if err != nil {
		fmt.Printf("エラー: 設定ファイルの読み込みに失敗しました: %v\n", err)
		return nil, nil, false
	}

	if len(cfg.APISources) == 0 {
		fmt.Println("エラー: api_sources が設定されていません。")
		return nil, nil, false
	}

	if cfg.AIAPIKey == "" {
		fmt.Println("エラー: APIキーが設定されていません。")
		return nil, nil, false
	}

	provider, err := ai.NewProvider(cfg.AIProvider, cfg.AIAPIKey)
	if err != nil {
		fmt.Printf("エラー: AIプロバイダーの作成に失敗しました: %v\n", err)
		return nil, nil, false
	}

	return cfg, provider, true
}

func runEndpoints(args []string) {
	// Parse common options
	commonOpts := parseCommonOptions(args)

	cfg, provider, ok := loadConfigAndProvider(commonOpts)
	if !ok {
		// Provide more detailed error message for api_sources if needed
		if cfg == nil {
			os.Exit(1)
		}
		if len(cfg.APISources) == 0 {
			fmt.Println("設定ファイルに以下のように api_sources を追加してください:")
			fmt.Println(`
api_sources:
  - type: express
    patterns:
      - "src/routes/**/*.ts"
  - type: openapi
    patterns:
      - "docs/openapi.yaml"`)
		}
		os.Exit(1)
	}

	if !commonOpts.jsonOutput {
		fmt.Println("\n📡 APIエンドポイントを抽出中...\n")
	}

	ctx := context.Background()
	endpoints, err := parser.ExtractEndpoints(ctx, cfg.APISources, provider)
	if err != nil {
		fmt.Printf("エラー: エンドポイントの抽出に失敗しました: %v\n", err)
		os.Exit(1)
	}

	if commonOpts.jsonOutput {
		outputEndpointsJSON(endpoints)
	} else {
		outputEndpointsConsole(endpoints)
	}
}

func outputEndpointsJSON(endpoints []parser.Endpoint) {
	data, _ := json.MarshalIndent(endpoints, "", "  ")
	fmt.Println(string(data))
}

func outputEndpointsConsole(endpoints []parser.Endpoint) {
	if len(endpoints) == 0 {
		fmt.Println("エンドポイントが見つかりませんでした。")
		return
	}

	fmt.Printf("📡 検出されたエンドポイント (%d件)\n", len(endpoints))
	fmt.Println(strings.Repeat("━", 60))

	// ソースごとにグループ化
	bySource := make(map[string][]parser.Endpoint)
	for _, ep := range endpoints {
		bySource[ep.Source] = append(bySource[ep.Source], ep)
	}

	for source, eps := range bySource {
		fmt.Printf("\n📁 %s (%d件)\n", source, len(eps))
		fmt.Println(strings.Repeat("─", 40))
		for _, ep := range eps {
			desc := ""
			if ep.Description != "" {
				desc = fmt.Sprintf(" - %s", ep.Description)
			}
			file := ""
			if ep.File != "" {
				file = fmt.Sprintf(" [%s]", ep.File)
			}
			fmt.Printf("  %-7s %s%s%s\n", ep.Method, ep.Path, desc, file)
		}
	}

	fmt.Println()
}

func runCoverage(args []string) {
	// Parse common options
	commonOpts := parseCommonOptions(args)

	cfg, provider, ok := loadConfigAndProvider(commonOpts)
	if !ok {
		// Provide more specific error message for coverage command
		if cfg != nil && len(cfg.APISources) == 0 {
			fmt.Println("カバレッジレポートにはAPIエンドポイントの抽出設定が必要です。")
		}
		os.Exit(1)
	}

	if !commonOpts.jsonOutput {
		fmt.Println("\n📊 APIカバレッジレポートを生成中...\n")
	}

	ctx := context.Background()
	report, err := parser.CalculateCoverage(ctx, cfg, provider)
	if err != nil {
		fmt.Printf("エラー: カバレッジレポートの生成に失敗しました: %v\n", err)
		os.Exit(1)
	}

	if commonOpts.jsonOutput {
		outputCoverageJSON(report)
	} else {
		outputCoverageConsole(report)
	}
}

func outputCoverageJSON(report *parser.CoverageReport) {
	data, _ := json.MarshalIndent(report, "", "  ")
	fmt.Println(string(data))
}

func outputCoverageConsole(report *parser.CoverageReport) {
	fmt.Println(strings.Repeat("━", 60))
	fmt.Println("📊 APIカバレッジレポート")
	fmt.Println(strings.Repeat("━", 60))

	// カバレッジサマリー
	emoji := getStatusEmoji(report.CoveragePercentage)
	fmt.Printf("\n%s カバレッジ: %.1f%%\n", emoji, report.CoveragePercentage)
	fmt.Printf("   エンドポイント総数: %d\n", report.TotalEndpoints)
	fmt.Printf("   カバー済み (SPECあり): %d\n", report.CoveredEndpoints)
	fmt.Printf("   未カバー (SPECなし): %d\n", report.UncoveredEndpoints)
	fmt.Printf("   SPEC総数: %d\n", report.TotalSpecs)
	if report.OrphanedSpecs > 0 {
		fmt.Printf("   孤立SPEC (対応なし): %d\n", report.OrphanedSpecs)
	}

	// プログレスバー
	barLen := 30
	covered := int(report.CoveragePercentage / 100 * float64(barLen))
	if covered > barLen {
		covered = barLen
	}
	bar := strings.Repeat("█", covered) + strings.Repeat("░", barLen-covered)
	fmt.Printf("\n   [%s] %.1f%%\n", bar, report.CoveragePercentage)

	// カバー済みエンドポイント
	if len(report.Covered) > 0 {
		fmt.Printf("\n✅ カバー済みエンドポイント (%d件)\n", len(report.Covered))
		fmt.Println(strings.Repeat("─", 40))
		for _, item := range report.Covered {
			specInfo := ""
			if item.SpecFile != "" {
				specInfo = fmt.Sprintf(" → %s", item.SpecFile)
			}
			fmt.Printf("  %-7s %s%s\n", item.Method, item.Path, specInfo)
		}
	}

	// 未カバーエンドポイント
	if len(report.Uncovered) > 0 {
		fmt.Printf("\n❌ 未カバーエンドポイント (%d件)\n", len(report.Uncovered))
		fmt.Println(strings.Repeat("─", 40))
		for _, item := range report.Uncovered {
			file := ""
			if item.File != "" {
				file = fmt.Sprintf(" [%s]", item.File)
			}
			fmt.Printf("  %-7s %s%s\n", item.Method, item.Path, file)
		}
	}

	// 孤立したSPEC
	if len(report.Orphaned) > 0 {
		fmt.Printf("\n⚠️  孤立SPEC（対応するエンドポイントなし） (%d件)\n", len(report.Orphaned))
		fmt.Println(strings.Repeat("─", 40))
		for _, item := range report.Orphaned {
			routePath := ""
			if item.RoutePath != "" {
				routePath = fmt.Sprintf(" [%s]", item.RoutePath)
			}
			fmt.Printf("  📄 %s%s\n", item.File, routePath)
			if item.Title != "" {
				fmt.Printf("     %s\n", item.Title)
			}
		}
	}

	fmt.Println()
}
