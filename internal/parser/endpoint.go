package parser

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"strings"

	"github.com/k-totani/gh-spec-verify/internal/ai"
	"github.com/k-totani/gh-spec-verify/internal/config"
)

// Endpoint はAPIエンドポイントを表す
type Endpoint struct {
	// HTTPメソッド (GET, POST, PUT, DELETE, PATCH, GRAPHQL等)
	Method string `json:"method"`

	// パス (/users/:id など)
	Path string `json:"path"`

	// ソースタイプ (express, openapi, auto等)
	Source string `json:"source"`

	// 元ファイルパス
	File string `json:"file"`

	// 説明（あれば）
	Description string `json:"description,omitempty"`
}

// ExtractEndpoints は設定に基づいてエンドポイントを抽出する
func ExtractEndpoints(ctx context.Context, sources []config.APISource, provider ai.Provider) ([]Endpoint, error) {
	var allEndpoints []Endpoint

	for _, source := range sources {
		var endpoints []Endpoint
		var err error

		switch source.Type {
		case "openapi":
			endpoints, err = extractFromOpenAPI(source.Patterns)
		case "express", "fastify", "go-echo", "go-gin", "rails", "django", "graphql", "auto":
			endpoints, err = extractWithAI(ctx, source, provider)
		default:
			return nil, fmt.Errorf("unknown api source type: %s", source.Type)
		}

		if err != nil {
			return nil, fmt.Errorf("failed to extract endpoints from %s: %w", source.Type, err)
		}

		allEndpoints = append(allEndpoints, endpoints...)
	}

	return allEndpoints, nil
}

// extractFromOpenAPI はOpenAPI/Swaggerファイルからエンドポイントを抽出する
func extractFromOpenAPI(patterns []string) ([]Endpoint, error) {
	var endpoints []Endpoint

	for _, pattern := range patterns {
		matches, err := filepath.Glob(pattern)
		if err != nil {
			return nil, err
		}

		for _, file := range matches {
			eps, err := parseOpenAPIFile(file)
			if err != nil {
				return nil, fmt.Errorf("failed to parse %s: %w", file, err)
			}
			endpoints = append(endpoints, eps...)
		}
	}

	return endpoints, nil
}

// parseOpenAPIFile はOpenAPIファイルを解析する
func parseOpenAPIFile(filePath string) ([]Endpoint, error) {
	content, err := os.ReadFile(filePath)
	if err != nil {
		return nil, err
	}

	var endpoints []Endpoint

	// YAMLまたはJSON形式のOpenAPIを簡易パース
	// paths セクションから抽出
	contentStr := string(content)

	// 簡易的なパス抽出（正規表現ベース）
	// 本格的な実装ではopenapi3パーサーを使う
	pathPattern := regexp.MustCompile(`(?m)^\s{2}(/[^:\s]+):`)

	pathMatches := pathPattern.FindAllStringSubmatch(contentStr, -1)
	if len(pathMatches) == 0 {
		// JSON形式の場合
		pathPattern = regexp.MustCompile(`"(/[^"]+)":\s*\{`)
		pathMatches = pathPattern.FindAllStringSubmatch(contentStr, -1)
	}

	for _, pm := range pathMatches {
		if len(pm) < 2 {
			continue
		}
		path := pm[1]

		// パスごとにメソッドを探す
		// 簡易実装: 一般的なHTTPメソッドをすべて候補にする
		methods := []string{"get", "post", "put", "delete", "patch"}
		for _, method := range methods {
			// 簡易チェック: パスの近くにメソッドがあるか
			if strings.Contains(contentStr, path) {
				methodCheck := regexp.MustCompile(fmt.Sprintf(`"%s":\s*\{[^}]*"%s"`, regexp.QuoteMeta(path), method))
				yamlCheck := regexp.MustCompile(fmt.Sprintf(`%s:[\s\S]*?%s:`, regexp.QuoteMeta(path), method))
				if methodCheck.MatchString(contentStr) || yamlCheck.MatchString(contentStr) {
					endpoints = append(endpoints, Endpoint{
						Method: strings.ToUpper(method),
						Path:   path,
						Source: "openapi",
						File:   filePath,
					})
				}
			}
		}
	}

	// エンドポイントが見つからなかった場合、全メソッドをデフォルトで追加
	if len(endpoints) == 0 && len(pathMatches) > 0 {
		for _, pm := range pathMatches {
			if len(pm) >= 2 {
				endpoints = append(endpoints, Endpoint{
					Method: "GET",
					Path:   pm[1],
					Source: "openapi",
					File:   filePath,
				})
			}
		}
	}

	return endpoints, nil
}

// extractWithAI はAIを使ってエンドポイントを抽出する
func extractWithAI(ctx context.Context, source config.APISource, provider ai.Provider) ([]Endpoint, error) {
	var allEndpoints []Endpoint

	// パターンにマッチするファイルを収集
	var files []string
	for _, pattern := range source.Patterns {
		matches, err := filepath.Glob(pattern)
		if err != nil {
			// globが失敗した場合、再帰的なパターンかもしれない
			matches, err = findFilesRecursive(pattern)
			if err != nil {
				continue
			}
		}
		files = append(files, matches...)
	}

	if len(files) == 0 {
		return nil, nil
	}

	// ファイル内容を読み込む
	var fileContents []string
	for _, file := range files {
		content, err := os.ReadFile(file)
		if err != nil {
			continue
		}
		fileContents = append(fileContents, fmt.Sprintf("=== File: %s ===\n%s", file, string(content)))
	}

	if len(fileContents) == 0 {
		return nil, nil
	}

	// AIでエンドポイントを抽出
	aiResults, err := provider.ExtractEndpoints(ctx, source.Type, strings.Join(fileContents, "\n\n"))
	if err != nil {
		return nil, err
	}

	// ai.EndpointResult を parser.Endpoint に変換
	for _, result := range aiResults {
		ep := Endpoint{
			Method:      result.Method,
			Path:        result.Path,
			Source:      source.Type,
			File:        result.File,
			Description: result.Description,
		}
		if ep.Source == "" {
			ep.Source = source.Type
		}
		allEndpoints = append(allEndpoints, ep)
	}

	return allEndpoints, nil
}

// findFilesRecursive は再帰的にファイルを検索する（**パターン対応）
func findFilesRecursive(pattern string) ([]string, error) {
	var files []string

	// **/ を含むパターンを処理
	if strings.Contains(pattern, "**") {
		parts := strings.SplitN(pattern, "**", 2)
		baseDir := strings.TrimSuffix(parts[0], "/")
		if baseDir == "" {
			baseDir = "."
		}

		suffix := ""
		if len(parts) > 1 {
			suffix = strings.TrimPrefix(parts[1], "/")
		}

		err := filepath.Walk(baseDir, func(path string, info os.FileInfo, err error) error {
			if err != nil {
				return nil // エラーを無視して続行
			}
			if info.IsDir() {
				return nil
			}

			// サフィックスパターンにマッチするかチェック
			if suffix != "" {
				matched, _ := filepath.Match(suffix, filepath.Base(path))
				if !matched {
					// 拡張子でのマッチも試行
					ext := filepath.Ext(path)
					suffixExt := filepath.Ext(suffix)
					if ext != suffixExt {
						return nil
					}
				}
			}

			files = append(files, path)
			return nil
		})

		if err != nil {
			return nil, err
		}
	}

	return files, nil
}

var (
	bracesPathParamRegex       = regexp.MustCompile(`\{([^}]+)\}`)
	angleBracketPathParamRegex = regexp.MustCompile(`<[^:>]*:?([^>]+)>`)
)

// NormalizePath はパスを正規化する（:id, {id}, <id> を統一）
func NormalizePath(path string) string {
	// {id} -> :id
	path = bracesPathParamRegex.ReplaceAllString(path, ":$1")
	// <type:id> -> :id
	path = angleBracketPathParamRegex.ReplaceAllString(path, ":$1")
	return path
}
