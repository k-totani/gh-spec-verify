package parser

import (
	"bufio"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"strings"
)

// Spec はSPECファイルの解析結果を表す
type Spec struct {
	// ファイルパス
	FilePath string

	// SPECのタイプ (ui, api など)
	Type string

	// タイトル
	Title string

	// ルートパス（UIの場合）またはエンドポイント（APIの場合）
	RoutePath string

	// 関連ファイルのパス
	RelatedFiles []string

	// SPECの全文
	Content string

	// メタデータ（テーブルから抽出）
	Metadata map[string]string

	// セクション
	Sections map[string]string
}

// ParseSpec はSPECファイルを解析する
func ParseSpec(filePath string) (*Spec, error) {
	content, err := os.ReadFile(filePath)
	if err != nil {
		return nil, fmt.Errorf("failed to read spec file: %w", err)
	}

	spec := &Spec{
		FilePath:     filePath,
		Content:      string(content),
		RelatedFiles: []string{},
		Metadata:     make(map[string]string),
		Sections:     make(map[string]string),
	}

	// ファイルパスからタイプを推測
	spec.Type = inferSpecType(filePath)

	// 解析
	lines := strings.Split(string(content), "\n")
	spec.parseTitle(lines)
	spec.parseMetadataTable(lines)
	spec.parseRelatedFiles(string(content))
	spec.parseSections(lines)

	return spec, nil
}

// inferSpecType はファイルパスからSPECタイプを推測する
func inferSpecType(filePath string) string {
	dir := filepath.Dir(filePath)
	base := filepath.Base(dir)

	switch base {
	case "ui", "pages", "components":
		return "ui"
	case "api", "routes", "endpoints":
		return "api"
	default:
		return "unknown"
	}
}

// parseTitle はタイトルを解析する
func (s *Spec) parseTitle(lines []string) {
	for _, line := range lines {
		if strings.HasPrefix(line, "# ") {
			s.Title = strings.TrimPrefix(line, "# ")
			return
		}
	}
	s.Title = filepath.Base(s.FilePath)
}

// parseMetadataTable はメタデータテーブルを解析する
func (s *Spec) parseMetadataTable(lines []string) {
	tableRegex := regexp.MustCompile(`^\|\s*([^|]+)\s*\|\s*([^|]+)\s*\|$`)

	inTable := false
	for _, line := range lines {
		line = strings.TrimSpace(line)

		// テーブル開始を検出
		if strings.HasPrefix(line, "|") && strings.HasSuffix(line, "|") {
			inTable = true
		} else if inTable && line == "" {
			inTable = false
		}

		if !inTable {
			continue
		}

		// ヘッダー行やセパレーター行をスキップ
		if strings.Contains(line, "---") {
			continue
		}

		matches := tableRegex.FindStringSubmatch(line)
		if len(matches) == 3 {
			key := strings.TrimSpace(matches[1])
			value := strings.TrimSpace(matches[2])

			// バッククォートを除去
			value = strings.Trim(value, "`")

			s.Metadata[key] = value

			// 特定のキーを特別に処理
			switch key {
			case "パス", "Path", "path":
				s.RoutePath = value
			case "エンドポイント", "Endpoint", "endpoint":
				s.RoutePath = value
			}
		}
	}
}

// parseRelatedFiles は関連ファイルを解析する
func (s *Spec) parseRelatedFiles(content string) {
	// ~/path/to/file 形式を検出
	tildeRegex := regexp.MustCompile("`~/([^`]+)`")
	matches := tildeRegex.FindAllStringSubmatch(content, -1)
	for _, match := range matches {
		if len(match) == 2 {
			s.RelatedFiles = append(s.RelatedFiles, match[1])
		}
	}

	// src/path/to/file 形式を検出
	srcRegex := regexp.MustCompile("`(src/[^`]+)`")
	matches = srcRegex.FindAllStringSubmatch(content, -1)
	for _, match := range matches {
		if len(match) == 2 {
			s.RelatedFiles = append(s.RelatedFiles, match[1])
		}
	}
}

// parseSections はセクションを解析する
func (s *Spec) parseSections(lines []string) {
	currentSection := ""
	var sectionContent strings.Builder

	for _, line := range lines {
		// ## で始まるセクションヘッダーを検出
		if strings.HasPrefix(line, "## ") {
			// 前のセクションを保存
			if currentSection != "" {
				s.Sections[currentSection] = strings.TrimSpace(sectionContent.String())
			}
			currentSection = strings.TrimPrefix(line, "## ")
			sectionContent.Reset()
		} else if currentSection != "" {
			sectionContent.WriteString(line)
			sectionContent.WriteString("\n")
		}
	}

	// 最後のセクションを保存
	if currentSection != "" {
		s.Sections[currentSection] = strings.TrimSpace(sectionContent.String())
	}
}

// FindSpecFiles は指定ディレクトリ内のSPECファイルを検索する
func FindSpecFiles(specsDir string, specType string) ([]string, error) {
	var files []string

	searchDir := specsDir
	if specType != "" {
		searchDir = filepath.Join(specsDir, specType)
	}

	err := filepath.Walk(searchDir, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}

		if !info.IsDir() && strings.HasSuffix(path, ".md") {
			files = append(files, path)
		}

		return nil
	})

	if err != nil {
		// ディレクトリが存在しない場合は空のリストを返す
		if os.IsNotExist(err) {
			return []string{}, nil
		}
		return nil, fmt.Errorf("failed to walk specs directory: %w", err)
	}

	return files, nil
}

// FindCodeFilesWithCodePaths はSPECに関連するコードファイルを検索する（複数ベースディレクトリ対応）
func FindCodeFilesWithCodePaths(spec *Spec, codeDir string, codePaths []string) ([]string, error) {
	var files []string
	seen := make(map[string]bool)

	addFile := func(path string) {
		if !seen[path] {
			if _, err := os.Stat(path); err == nil {
				files = append(files, path)
				seen[path] = true
			}
		}
	}

	// codePaths が空の場合は codeDir をデフォルトとして使用
	if len(codePaths) == 0 {
		codePaths = []string{codeDir}
	}

	// ルートパスから推測
	if spec.RoutePath != "" {
		// /generators/synthesize -> synthesize
		routeName := filepath.Base(spec.RoutePath)
		if routeName == "" || routeName == "/" {
			routeName = "index"
		}

		// 各 codePath に対してパターンを試す
		for _, baseDir := range codePaths {
			// 可能なファイル名パターン
			patterns := []string{
				filepath.Join(baseDir, routeName+".tsx"),
				filepath.Join(baseDir, routeName+".ts"),
				filepath.Join(baseDir, routeName+".jsx"),
				filepath.Join(baseDir, routeName+".js"),
				filepath.Join(baseDir, "pages", routeName+".tsx"),
				filepath.Join(baseDir, "routes", routeName+".tsx"),
			}

			for _, pattern := range patterns {
				addFile(pattern)
			}

			// ディレクトリ内を検索
			if _, err := os.Stat(baseDir); err == nil {
				filepath.Walk(baseDir, func(path string, info os.FileInfo, err error) error {
					if err != nil {
						return nil
					}
					if info.IsDir() {
						return nil
					}

					// ファイル名にルート名が含まれているか
					baseName := strings.ToLower(filepath.Base(path))
					if strings.Contains(baseName, strings.ToLower(routeName)) {
						// テストファイルを除外
						if !strings.Contains(baseName, ".test.") && !strings.Contains(baseName, ".spec.") {
							addFile(path)
						}
					}

					return nil
				})
			}
		}
	}

	// 関連ファイルを追加（パス正規化で二重結合を防ぐ）
	for _, relFile := range spec.RelatedFiles {
		// resolveRelatedPath で正規化
		resolvedPath := resolveRelatedPath(codeDir, relFile)
		possiblePaths := []string{
			resolvedPath,
			resolvedPath + ".tsx",
			resolvedPath + ".ts",
		}
		for _, p := range possiblePaths {
			addFile(p)
		}
	}

	return files, nil
}

// FindCodeFiles はSPECに関連するコードファイルを検索する（後方互換用）
func FindCodeFiles(spec *Spec, codeDir string, mapping map[string]string) ([]string, error) {
	// マッピングからベースディレクトリを構築
	var codePaths []string
	if mapped, ok := mapping[spec.Type]; ok {
		codePaths = []string{filepath.Join(codeDir, mapped)}
	} else {
		codePaths = []string{codeDir}
	}

	// 新関数に委譲
	return FindCodeFilesWithCodePaths(spec, codeDir, codePaths)
}

// ReadFiles は複数のファイルを読み込む
func ReadFiles(paths []string) (map[string]string, error) {
	contents := make(map[string]string)

	for _, path := range paths {
		data, err := os.ReadFile(path)
		if err != nil {
			continue // エラーは無視して続行
		}
		contents[path] = string(data)
	}

	return contents, nil
}

// Scanner は標準入力からの読み取り用
func Scanner() *bufio.Scanner {
	return bufio.NewScanner(os.Stdin)
}

// resolveRelatedPath は関連ファイルパスを正規化する
// codeDir で始まるパスの二重結合を防ぐ
func resolveRelatedPath(codeDir, relFile string) string {
	// 絶対パスならそのまま返す
	if filepath.IsAbs(relFile) {
		return relFile
	}

	// 両方を正規化して比較
	cleanCodeDir := filepath.Clean(codeDir)
	cleanRelFile := filepath.Clean(relFile)

	// relFile が既に codeDir で始まる場合は結合しない
	// 例: codeDir=src, relFile=src/client/... → src/client/... をそのまま返す
	if strings.HasPrefix(cleanRelFile, cleanCodeDir+string(filepath.Separator)) || cleanRelFile == cleanCodeDir {
		return cleanRelFile
	}

	// それ以外は結合
	return filepath.Join(codeDir, relFile)
}
