package verifier

import (
	"context"
	"fmt"
	"path/filepath"
	"sync"

	"github.com/k-totani/gh-spec-verify/internal/ai"
	"github.com/k-totani/gh-spec-verify/internal/config"
	"github.com/k-totani/gh-spec-verify/internal/parser"
)

// Result は単一のSPEC検証結果
type Result struct {
	// SPECファイルのパス
	SpecFile string

	// SPECのタイトル
	Title string

	// ルートパス
	RoutePath string

	// 見つかったコードファイル
	CodeFiles []string

	// 検証結果
	Verification *ai.VerificationResult

	// エラー（検証に失敗した場合）
	Error error
}

// FailingSpec は個別閾値を下回ったSPECの情報
type FailingSpec struct {
	// SPECファイルのパス
	SpecFile string `json:"specFile"`

	// SPECのタイトル
	Title string `json:"title"`

	// 一致度（パーセント）
	MatchPercentage int `json:"matchPercentage"`
}

// Summary は全体の検証サマリー
type Summary struct {
	// 総SPEC数
	TotalSpecs int

	// 検証成功数
	VerifiedSpecs int

	// 平均一致度
	AverageMatch float64

	// 高一致数（80%以上）
	HighMatchCount int

	// 低一致数（50%未満）
	LowMatchCount int

	// 個別結果
	Results []Result

	// 個別閾値（この値未満は失敗）
	FailUnder int `json:"failUnder,omitempty"`

	// 個別閾値を下回ったSPEC一覧
	FailingSpecs []FailingSpec `json:"failingSpecs,omitempty"`
}

// Verifier はSPEC検証を行う
type Verifier struct {
	config   *config.Config
	provider ai.Provider
}

// New は新しいVerifierを作成する
func New(cfg *config.Config) (*Verifier, error) {
	provider, err := ai.NewProvider(cfg.AIProvider, cfg.AIAPIKey)
	if err != nil {
		return nil, fmt.Errorf("failed to create AI provider: %w", err)
	}

	return &Verifier{
		config:   cfg,
		provider: provider,
	}, nil
}

// VerifyAll は全てのSPECを検証する
// specTypeが空の場合は全タイプ、指定がある場合はそのタイプのみを検証する
func (v *Verifier) VerifyAll(ctx context.Context, specType string) (*Summary, error) {
	// specTypeが指定されている場合は単一タイプ、空の場合は全タイプを検証
	var specTypes []string
	if specType == "" {
		// 全タイプを検証（空文字列を渡すとparser.FindSpecFilesが全タイプを検索する）
		specTypes = []string{""}
	} else {
		specTypes = []string{specType}
	}

	return v.VerifyMultipleTypes(ctx, specTypes)
}

// VerifyOne は単一のSPECを検証する
func (v *Verifier) VerifyOne(ctx context.Context, specFile string) (*Result, error) {
	result := v.verifyOne(ctx, specFile)
	if result.Error != nil {
		return nil, result.Error
	}
	return &result, nil
}

// verifyOne は単一のSPECを検証する（内部用）
func (v *Verifier) verifyOne(ctx context.Context, specFile string) Result {
	result := Result{
		SpecFile: filepath.Base(specFile),
	}

	// SPECファイルを解析
	spec, err := parser.ParseSpec(specFile)
	if err != nil {
		result.Error = fmt.Errorf("failed to parse spec: %w", err)
		return result
	}

	result.Title = spec.Title
	result.RoutePath = spec.RoutePath

	// 関連コードファイルを検索
	codeFiles, err := parser.FindCodeFiles(spec, v.config.CodeDir, v.config.Mapping)
	if err != nil {
		result.Error = fmt.Errorf("failed to find code files: %w", err)
		return result
	}

	result.CodeFiles = codeFiles

	if len(codeFiles) == 0 {
		result.Verification = &ai.VerificationResult{
			MatchPercentage: 0,
			MatchedItems:    []string{},
			UnmatchedItems:  []string{"対応するコードが見つかりません"},
			Notes:           "未実装の可能性があります",
		}
		return result
	}

	// コードファイルを読み込む
	codeContents, err := parser.ReadFiles(codeFiles)
	if err != nil {
		result.Error = fmt.Errorf("failed to read code files: %w", err)
		return result
	}

	// 検証観点を取得
	verificationFocus := v.config.GetVerificationFocus(spec.Type)

	// AIで検証
	var verification *ai.VerificationResult
	if len(verificationFocus) > 0 {
		opts := &ai.VerifyOptions{
			VerificationFocus: verificationFocus,
		}
		verification, err = v.provider.VerifyWithOptions(ctx, spec.Content, codeContents, opts)
	} else {
		verification, err = v.provider.Verify(ctx, spec.Content, codeContents)
	}
	if err != nil {
		result.Error = fmt.Errorf("failed to verify with AI: %w", err)
		return result
	}

	result.Verification = verification
	return result
}

// calculateSummary はサマリーを計算する
func (v *Verifier) calculateSummary(results []Result) *Summary {
	summary := &Summary{
		TotalSpecs: len(results),
		Results:    results,
	}

	var totalMatch int
	for _, result := range results {
		if result.Error == nil && result.Verification != nil {
			summary.VerifiedSpecs++
			totalMatch += result.Verification.MatchPercentage

			if result.Verification.MatchPercentage >= 80 {
				summary.HighMatchCount++
			} else if result.Verification.MatchPercentage < 50 {
				summary.LowMatchCount++
			}
		}
	}

	if summary.VerifiedSpecs > 0 {
		summary.AverageMatch = float64(totalMatch) / float64(summary.VerifiedSpecs)
	}

	return summary
}

// IsPassing は検証が合格基準を満たしているかを返す
func (s *Summary) IsPassing(threshold int) bool {
	return s.AverageMatch >= float64(threshold)
}

// VerifyMultipleTypes は複数のSPECタイプを検証する
func (v *Verifier) VerifyMultipleTypes(ctx context.Context, specTypes []string) (*Summary, error) {
	var allResults []Result

	for _, specType := range specTypes {
		// SPECファイルを検索
		specFiles, err := parser.FindSpecFiles(v.config.SpecsDir, specType)
		if err != nil {
			return nil, fmt.Errorf("failed to find spec files for type %s: %w", specType, err)
		}

		if len(specFiles) == 0 {
			continue
		}

		// 結果を格納するチャネル
		resultChan := make(chan Result, len(specFiles))

		// 並列実行のためのワーカープール
		var wg sync.WaitGroup
		semaphore := make(chan struct{}, v.config.Options.Concurrency)

		for _, specFile := range specFiles {
			wg.Add(1)
			go func(sf string) {
				defer wg.Done()
				semaphore <- struct{}{}        // 取得
				defer func() { <-semaphore }() // 解放

				result := v.verifyOne(ctx, sf)
				resultChan <- result
			}(specFile)
		}

		// 全ての検証が完了するのを待つ
		go func() {
			wg.Wait()
			close(resultChan)
		}()

		// 結果を収集
		for result := range resultChan {
			allResults = append(allResults, result)
		}
	}

	if len(allResults) == 0 {
		return &Summary{
			TotalSpecs: 0,
			Results:    []Result{},
		}, nil
	}

	// サマリーを計算
	return v.calculateSummary(allResults), nil
}
