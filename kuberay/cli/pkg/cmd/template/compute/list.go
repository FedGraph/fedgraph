package compute

import (
	"context"
	"log"
	"os"
	"strconv"
	"time"

	"github.com/olekukonko/tablewriter"
	"github.com/ray-project/kuberay/cli/pkg/cmdutil"
	"github.com/ray-project/kuberay/proto/go_client"
	"github.com/spf13/cobra"
)

type ListOptions struct {
	namespace string
}

func newCmdList() *cobra.Command {
	opts := ListOptions{}

	cmd := &cobra.Command{
		Use:   "list",
		Short: "List all compute templates",
		Args:  cobra.NoArgs,
		RunE: func(cmd *cobra.Command, args []string) error {
			return listComputeTemplates(opts)
		},
	}

	cmd.Flags().StringVarP(&opts.namespace, "namespace", "n", "",
		"kubernetes namespace where the compute template is stored")

	return cmd
}

func listComputeTemplates(opts ListOptions) error {
	// Get gRPC connection
	conn, err := cmdutil.GetGrpcConn()
	if err != nil {
		return err
	}
	defer conn.Close()

	// build gRPC client
	client := go_client.NewComputeTemplateServiceClient(conn)
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	var computeTemplates []*go_client.ComputeTemplate
	if len(opts.namespace) == 0 {
		r, err := client.ListAllComputeTemplates(ctx, &go_client.ListAllComputeTemplatesRequest{})
		if err != nil {
			log.Fatalf("could not list all compute templates %v", err)
		}
		computeTemplates = r.GetComputeTemplates()
	} else {
		r, err := client.ListComputeTemplates(ctx, &go_client.ListComputeTemplatesRequest{
			Namespace: opts.namespace,
		})
		if err != nil {
			log.Fatalf("could not list compute templates %v", err)
		}
		computeTemplates = r.GetComputeTemplates()
	}
	rows := convertComputeTemplatesToStrings(computeTemplates)

	table := tablewriter.NewWriter(os.Stdout)
	table.SetHeader([]string{"Name", "Namespace", "CPU", "Memory", "GPU", "GPU-Accelerator"})
	table.AppendBulk(rows)
	table.Render()

	return nil
}

func convertComputeTemplatesToStrings(computeTemplates []*go_client.ComputeTemplate) [][]string {
	var data [][]string

	for _, r := range computeTemplates {
		data = append(data, convertComputeTemplatToString(r))
	}

	return data
}

func convertComputeTemplatToString(r *go_client.ComputeTemplate) []string {
	line := []string{
		r.GetName(), r.Namespace, strconv.Itoa(int(r.GetCpu())), strconv.Itoa(int(r.Memory)),
		strconv.Itoa(int(r.GetGpu())), r.GetGpuAccelerator(),
	}
	return line
}
