package parser

import (
	"context"
	"path/filepath"
	"strings"

	"github.com/k-totani/spec-verify/internal/ai"
	"github.com/k-totani/spec-verify/internal/config"
)

// CoverageReport はAPIエンドポイントのカバレッジレポート
type CoverageReport struct {
	// 総エンドポイント数
	TotalEndpoints int `json:"totalEndpoints"`

	// カバーされているエンドポイント数（SPECあり）
	CoveredEndpoints int `json:"coveredEndpoints"`

	// カバーされていないエンドポイント数（SPECなし）
	UncoveredEndpoints int `json:"uncoveredEndpoints"`

	// カバレッジ率（パーセント）
	CoveragePercentage float64 `json:"coveragePercentage"`

	// 総SPEC数
	TotalSpecs int `json:"totalSpecs"`

	// 孤立したSPEC数（エンドポイントなし）
	OrphanedSpecs int `json:"orphanedSpecs"`

	// カバーされているエンドポイント詳細
	Covered []CoverageItem `json:"covered"`

	// カバーされていないエンドポイント詳細
	Uncovered []CoverageItem `json:"uncovered"`

	// 孤立したSPEC詳細（対応するエンドポイントがないSPEC）
	Orphaned []OrphanedSpec `json:"orphaned,omitempty"`
}

// CoverageItem はカバレッジ項目
type CoverageItem struct {
	// HTTPメソッド
	Method string `json:"method"`

	// エンドポイントパス
	Path string `json:"path"`

	// ソースタイプ
	Source string `json:"source"`

	// 元ファイル
	File string `json:"file,omitempty"`

	// 対応するSPECファイル（カバーされている場合）
	SpecFile string `json:"specFile,omitempty"`
}

// OrphanedSpec は孤立したSPEC（対応するエンドポイントがない）
type OrphanedSpec struct {
	// SPECファイルパス
	File string `json:"file"`

	// タイトル
	Title string `json:"title"`

	// SPECに記載されたパス
	RoutePath string `json:"routePath,omitempty"`
}

// CalculateCoverage はエンドポイントとSPECのカバレッジを計算する
func CalculateCoverage(ctx context.Context, cfg *config.Config, provider ai.Provider) (*CoverageReport, error) {
	report := &CoverageReport{
		Covered:   []CoverageItem{},
		Uncovered: []CoverageItem{},
		Orphaned:  []OrphanedSpec{},
	}

	// エンドポイントを抽出
	endpoints, err := ExtractEndpoints(ctx, cfg.APISources, provider)
	if err != nil {
		return nil, err
	}
	report.TotalEndpoints = len(endpoints)

	// SPECファイルを検索（APIタイプのみ）
	specFiles, err := FindSpecFiles(cfg.SpecsDir, "api")
	if err != nil {
		return nil, err
	}
	// apiディレクトリがなければ全SPECを検索
	if len(specFiles) == 0 {
		specFiles, err = FindSpecFiles(cfg.SpecsDir, "")
		if err != nil {
			return nil, err
		}
	}
	report.TotalSpecs = len(specFiles)

	// SPECをパースしてルートパスを取得
	specs := make([]*Spec, 0, len(specFiles))
	specPathMap := make(map[string]*Spec) // 正規化パス -> Spec

	for _, specFile := range specFiles {
		spec, err := ParseSpec(specFile)
		if err != nil {
			continue
		}
		specs = append(specs, spec)

		// ルートパスがある場合はマップに追加
		if spec.RoutePath != "" {
			normalizedPath := NormalizePath(spec.RoutePath)
			specPathMap[normalizedPath] = spec
		}
	}

	// マッチング用のセット
	matchedSpecs := make(map[string]bool)

	// 各エンドポイントをチェック
	for _, ep := range endpoints {
		normalizedPath := NormalizePath(ep.Path)
		item := CoverageItem{
			Method: ep.Method,
			Path:   ep.Path,
			Source: ep.Source,
			File:   ep.File,
		}

		// SPECとマッチするか確認
		if spec, found := specPathMap[normalizedPath]; found {
			item.SpecFile = filepath.Base(spec.FilePath)
			report.Covered = append(report.Covered, item)
			report.CoveredEndpoints++
			matchedSpecs[spec.FilePath] = true
		} else {
			// パスの一部マッチも試行
			matched := false
			for specPath, spec := range specPathMap {
				if pathsMatch(normalizedPath, specPath) {
					item.SpecFile = filepath.Base(spec.FilePath)
					report.Covered = append(report.Covered, item)
					report.CoveredEndpoints++
					matchedSpecs[spec.FilePath] = true
					matched = true
					break
				}
			}
			if !matched {
				report.Uncovered = append(report.Uncovered, item)
				report.UncoveredEndpoints++
			}
		}
	}

	// 孤立したSPECを検出
	for _, spec := range specs {
		if !matchedSpecs[spec.FilePath] {
			report.Orphaned = append(report.Orphaned, OrphanedSpec{
				File:      filepath.Base(spec.FilePath),
				Title:     spec.Title,
				RoutePath: spec.RoutePath,
			})
			report.OrphanedSpecs++
		}
	}

	// カバレッジ率を計算
	if report.TotalEndpoints > 0 {
		report.CoveragePercentage = float64(report.CoveredEndpoints) / float64(report.TotalEndpoints) * 100
	}

	return report, nil
}

// pathsMatch は2つのパスがマッチするか確認する
// 完全一致、またはパラメータ部分を除いた一致をチェック
func pathsMatch(path1, path2 string) bool {
	// Handle empty paths
	if path1 == "" || path2 == "" {
		return path1 == path2
	}

	// 完全一致
	if path1 == path2 {
		return true
	}

	// セグメントに分割
	segments1 := strings.Split(strings.Trim(path1, "/"), "/")
	segments2 := strings.Split(strings.Trim(path2, "/"), "/")

	// セグメント数が異なる場合は不一致
	if len(segments1) != len(segments2) {
		return false
	}

	// 各セグメントを比較
	for i := range segments1 {
		s1 := segments1[i]
		s2 := segments2[i]

		// どちらかがパラメータの場合はマッチとみなす
		if isPathParameter(s1) || isPathParameter(s2) {
			continue
		}

		// 通常のセグメントは完全一致が必要
		if s1 != s2 {
			return false
		}
	}

	return true
}

// isPathParameter はセグメントがパスパラメータかを判定
func isPathParameter(segment string) bool {
	if len(segment) < 2 {
		return false
	}
	// :id, {id}, <id> patterns need at least 2 chars
	return segment[0] == ':' ||
		(segment[0] == '{' && segment[len(segment)-1] == '}') ||
		(segment[0] == '<' && segment[len(segment)-1] == '>')
}
